# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from bids import BIDSLayout
from numpy.linalg import norm

from nilearn.glm import Contrast
from nilearn.maskers import NiftiMasker
from nilearn.image import resample_to_img
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.contrasts import compute_fixed_effects

from codecarbon import OfflineEmissionsTracker

# Specify the runs to exclude
EXCEPT_FILES = [
    {
        'subject': '03',
        'session': '001',
        'run': ['01', '02', '03']
    },
    {
        'subject': '05',
        'session': '002',
        'run': ['02', '04']
    }
]

# Specify which columns of the events files correspond to the `onset` and `duration`
MAPPING_KEY = {
    'onset': 'onset_video_flip',
    'duration': 'total_duration'
}

def check_bids_input(bids_input, input_type, match):
    if len(bids_input) == 0:
        raise FileNotFoundError(f"No file {input_type} associated with {match}")
    elif len(bids_input) > 1:
        raise ValueError(f"More than one {input_type} file associated with {match}")
    else:
        return bids_input[0]


def check_file_exist(file):
    if not Path(file).exists:
        raise FileNotFoundError(f"{file} not found")

def get_tr(layout_fmriprep, entities, match):
    # Retrieve file
    r_t = layout_fmriprep.get(subject=entities['subject'], session=entities['session'], run=entities['run'],  echo='2', desc='preproc', suffix='bold', extension='.nii.gz')

    # Validate input
    r_t = check_bids_input(r_t, 'fmriprep bold', match)
    
    return r_t.get_metadata()['RepetitionTime']


def get_events(layout_raw, entities, match):
    # Retrieve file
    events = layout_raw.get(subject=entities['subject'], session=entities['session'], run=entities['run'], suffix='events', extension='.tsv')

    # Validate input
    events = check_bids_input(events, 'events', match)
    
    return events.get_df()


def get_mask(entities, file, layout_mask=None, resample_mask=True, atlas=None):
    # Retrieve file
    if layout_mask is not None:
        mask = layout_mask.get(subject=entities['subject'], space='MNI152NLin2009cAsym', desc='brain', suffix='mask', extension='.nii.gz')
        # Validate input
        mask = check_bids_input(mask, 'MNI152Lin2009cAsym mask', file.filename)
        mask = mask.path
    elif atlas is not None:
        # Validate input
        check_file_exist(atlas)
        mask = atlas
        entities.update({
            'atlas': [entity.split('-')[1] for entity in atlas.split("_") if 'atlas-' in entity][0],
            'label': [entity.split('-')[1] for entity in atlas.split("_") if 'label-' in entity][0]
        })
    else:
        raise ValueError('Both `layout_mask` and `atlas` arguments set to None. Please specify one of those argument')

    # Resample mask to save
    if resample_mask:
        mask = resample_to_img(mask, file.path, interpolation='nearest')

    return mask, entities


def design_matrix_parametric(modulators, events, mapping_key, interaction=True, modulator_split=None):
    # Make sure `modulators` is a list
    modulators = [modulators] if not isinstance(modulators, list) else modulators
    # Check if modulators are valid
    if not set(modulators).issubset(events.columns): raise ValueError(f"The following modulator(s) was not found in the events file: {set(modulators) - set(events.columns)}")

    if modulator_split is not None:
        events = split_modulator(events, modulator_split)

    tmp_events, tmp_mean_centered, exp_regressors = [], [], []
    for modulator in modulators:
        mean_centered = np.array(events[modulator] - events[modulator].mean())
        tmp_mean_centered.append(mean_centered)

        if modulator_split is not None:
            list_t_type = events['trial_type'] + f'_{modulator}'
            tmp_events.append(
                pd.DataFrame({
                    'trial_type': events['trial_type'],
                    'onset': events[mapping_key['onset']],
                    'duration': events[mapping_key['duration']],
                    'modulation': [1]*len(events)
                })
            )
        else:
            list_t_type = [modulator]*len(events)

        # Create events structure
        tmp_events.append(
            pd.DataFrame({
                'trial_type': list_t_type,
                'onset': events[mapping_key['onset']],
                'duration': events[mapping_key['duration']],
                'modulation': mean_centered
            })
        )
        exp_regressors.append(modulator)
    if interaction and len(tmp_events)>1:
        # Add interaction
        interaction_term = "x".join(modulators)
        if modulator_split is not None:
            interaction_term = events['trial_type'] + f'_{interaction_term}'
        else:
            interaction_term = [interaction_term]*len(events)

        tmp_events.append(
            pd.DataFrame({
                'trial_type': interaction_term,
                'onset': events[mapping_key['onset']],
                'duration': events[mapping_key['duration']],
                'modulation': tmp_mean_centered[0]*tmp_mean_centered[1]
            })
        )
        exp_regressors.append(interaction_term)

    if len(tmp_events)>1:
        tmp_events = pd.concat(tmp_events)
        return tmp_events, tmp_events['trial_type'].unique().tolist()
    else:
        return tmp_events[0], exp_regressors


def split_modulator(events, modulator, threshold=5):
    trial_type = []
    for idx, row in events.iterrows():
        if row[modulator]>=threshold:
            trial_type.append('positive')
        elif row[modulator]<threshold:
            trial_type.append('negative')

    events['trial_type'] = trial_type

    return events


def run_first_level_glm(layout_tedana, layout_raw, layout_fmriprep, subject, layout_mask=None, atlas=None, desc='denoised', resample_mask=True, trial_type=None, modulator=None, interaction=False, modulator_split=None):
    
    logger = logging.getLogger(__name__)
    
    stats_imgs = {}
        
    # Retrieve bold data for given `desc`
    files = layout_tedana.get(subject=subject, desc=desc, space='MNI152NLin2009cAsym', suffix='bold', extension='.nii.gz')
    
    for except_file in EXCEPT_FILES:
        if except_file['subject'] == subject:
            files = [f for f in files if not (f.entities['session'] == except_file['session'] and f.entities['run'] in except_file['run'])]
            
    for file in files:
        logger.info(f"... running GLM for: {file.filename}")

        entities = file.get_entities()
        entities.update({
            'desc': desc
        })

        # Retrieve repetition time
        r_t = get_tr(layout_fmriprep, entities, file.filename)
    
        # Retrieve events file
        events = get_events(layout_raw, entities, file.filename)

        # Prepare events for design matrix
        if modulator is not None:
            events, exp_regressors = design_matrix_parametric(modulator, events, MAPPING_KEY, interaction=interaction, modulator_split=modulator_split)
        else:
            if trial_type == 'Gif':
                exp_regressors = events[trial_type].str.replace('.mp4', '', regex=False)
            if trial_type == 'category':
                exp_regressors = events['category'].to_list()
            else:
                exp_regressors = ['Stim']*len(events)

            events = pd.DataFrame({'trial_type': exp_regressors, 'onset': events[MAPPING_KEY['onset']], 'duration': events[MAPPING_KEY['duration']]})

        # Retrieve mask
        mask, entities = get_mask(entities, file, layout_mask=layout_mask, atlas=atlas, resample_mask=resample_mask)

        logger.info(f"... fitting the First level model with the following parameters: ")
        logger.info(f"...     mask: {mask}")
        logger.info(f"...     bold: {file.path}")
        logger.info(f"...     rt: {r_t}")
        logger.info(f"...     events: {events}")
        breakpoint()
        # First level GLM
        first_level_model = FirstLevelModel(r_t, mask_img=mask, n_jobs=-1, signal_scaling=(0, 1), smoothing_fwhm=5)
        fmri_glm = first_level_model.fit(file.path, events=events)
        
        # Define contrast
        design_matrix=fmri_glm.design_matrices_[0]
        n_regressors = design_matrix.shape[1]

        for idx, regressor in enumerate(list(set(exp_regressors))):
            logger.info(f"... computing first level contrast for regressor: {regressor}")
            activation = np.zeros(n_regressors)
            activation[idx] = 1

            contrast = fmri_glm.compute_contrast(activation, output_type='all')

            if regressor not in stats_imgs.keys():
                stats_imgs.update({
                    regressor: [contrast]
                })
            else:
                stats_imgs[regressor].append(contrast)
        
        if modulator_split is not None:
            conditions = modulator.copy()
            if interaction:
                conditions.append('x'.join(conditions))
            for condition in conditions:
                logger.info(f"... computing first level contrast for regressor: {condition}")
                activation_idx = [i for i, r in enumerate(exp_regressors) if r.split('_')[-1]==condition]
                activation = np.zeros(n_regressors)
                activation[activation_idx] = 1
                logger.info(f"...     regressors: {exp_regressors}")
                logger.info(f"...     contrast: {activation}")
                
                contrast = fmri_glm.compute_contrast(activation, output_type='all')

                if condition not in stats_imgs.keys():
                    stats_imgs.update({
                        condition: [contrast]
                    })
                else:
                    stats_imgs[condition].append(contrast)                

    return stats_imgs, entities


def run_second_level_glm(stats_imgs, entities, path_output):

    logger = logging.getLogger(__name__)

    # Update `desc` value to reflect model type
    tmp_entities = entities.copy()
    tmp_entities.update({
        'desc': ''.join([tmp_entities['desc'], 'rfx'])
    })

    pattern = "sub-{subject}/sub-{subject}_task-{task}_space-{space}[_atlas-{atlas}][_label-{label}]_contrast-contrast_stat-stat_desc-{desc}_statmap.nii.gz"
    
    for regressor in stats_imgs.keys():
        # Retrieve maps
        betas = [simg['effect_size'] for simg in stats_imgs[regressor]]
        # Run second level
        logger.info(f"... fitting the Second level model for the regressor: {regressor}")
        second_level_model = SecondLevelModel()
        second_level_model = second_level_model.fit(betas, design_matrix=pd.DataFrame({regressor: [1]*len(betas)}))
        # Compute contrast
        logger.info(f"... computing second level contrast for regressor: {regressor}")
        output_img = second_level_model.compute_contrast(
            second_level_contrast=regressor, second_level_stat_type='t', output_type='all'
        )
        # Save the output: z-score
        out_file = path_output.build_path(tmp_entities, pattern, validate=False).replace('contrast-contrast', f'contrast-{regressor}').replace('stat-stat', f'stat-z')
        logger.info(f"... saving the random effect zscore map : {out_file}")
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        output_img['z_score'].to_filename(out_file)
        # Save the output: effec size
        out_file = path_output.build_path(tmp_entities, pattern, validate=False).replace('contrast-contrast', f'contrast-{regressor}').replace('stat-stat', f'stat-effect')
        logger.info(f"... saving the random effect effect size map : {out_file}")
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        output_img['effect_size'].to_filename(out_file)


def run_fixed_effect(stats_imgs, entities, path_output):

    logger = logging.getLogger(__name__)

    # Update `desc` value to reflect model type
    tmp_entities = entities.copy()
    tmp_entities.update({
        'desc': ''.join([tmp_entities['desc'], 'ffx'])
    })

    pattern = "sub-{subject}/sub-{subject}_task-{task}_space-{space}[_atlas-{atlas}][_label-{label}]_contrast-contrast_stat-stat_desc-{desc}_statmap.nii.gz"
    
    for regressor in stats_imgs.keys():
        logger.info(f"... computing the fixed effect model for regressor: {regressor}")
        # Compute fixed effect
        ffx_contrast, ffx_variance, ffx_stat = compute_fixed_effects(
            [simg["effect_size"] for simg in stats_imgs[regressor]],
            [simg["effect_variance"] for simg in stats_imgs[regressor]]
        )

        # Save the output
        for stat_map, name in zip([ffx_contrast, ffx_variance, ffx_stat], ['contrast','variance', 'stat']):
            # Generate filename
            out_file = path_output.build_path(tmp_entities, pattern, validate=False).replace('contrast-contrast', f'contrast-{regressor}').replace('stat-stat', f'stat-{name}')

            logger.info(f"... saving the fixed effect {name} map : {out_file}")
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            stat_map.to_filename(out_file)


@click.command()
@click.argument('ds_tedana', type=str)
@click.argument('ds_raw', type=click.Path())
@click.argument('ds_fmriprep', type=click.Path())
@click.argument('path_output', type=click.Path())
@click.option('--ds_smriprep', type=click.Path(), default=None, help='Path containing the anatomical data')
@click.option('--atlas', type=click.Path(), default=None, help='File to use for ROI analysis. If not specify, will assumed that path to brain mask wwas specified in the `--ds_smriprep` arugment')
@click.option('--subject', type=str, default=None, help='Specify the subject id to run the analysis for a specific subject. Other, will run the analysis for all subjects found in the `ds_tedana` path')
@click.option('--desc', type=click.Choice(['denoised', 'optcom']), default='denoised', help='Specify which images to use for the anaylsis between denoised and optcom')
@click.option('--trial_type', type=str, default=None, help='')
@click.option('--resample_mask', is_flag=True, help='Flag to specify if XXX')
@click.option('--model', type=click.Choice(['fixed', 'random', 'both', None]), default='random', help='Specify the analysis type: fixed (fixed effect model), random (second level model) or both (consecutively)')
@click.option('--modulator', multiple=True, default=None, help='Parametric modulators to include in the analysis. The value specified for each modulator should match the name of a column in the events files in the `ds_raw` path')
@click.option('--interaction', is_flag=True, help='Specify flag to include an interaction terms between the parametric modulators specified in the `--modulator` argument')
@click.option('--modulator_split', type=str, default=None, help='')
def main(ds_tedana, ds_raw, ds_fmriprep, path_output, ds_smriprep, atlas, subject, desc, trial_type, resample_mask, model, modulator, interaction, modulator_split):

    # Reformat modulator input
    if modulator is not None:
        modulator = list(modulator)
        contrast = "".join(m.capitalize() for m in modulator)
    if len(modulator)==0:
        modulator=None
        contrast = trial_type

    # Setup carbon tracker
    tracker_dir = Path(path_output) / 'log'
    tracker_dir.mkdir(parents=True, exist_ok=True)
    tracker_file = f'emissions_model-{model}_contrast-{contrast}_desc-{desc}.csv'
    tracker = OfflineEmissionsTracker(
        output_dir=tracker_dir, output_file=tracker_file, country_iso_code='CAN'
    )
    tracker.start()

    # Defined BIDSLayouts
    layout_raw = BIDSLayout(ds_raw, validate=False, is_derivative=True)
    layout_fmriprep = BIDSLayout(ds_fmriprep, validate=False, is_derivative=True)
    layout_tedana = BIDSLayout(ds_tedana, validate=False, is_derivative=True)
    layout_smriprep = BIDSLayout(ds_smriprep, validate=False, is_derivative=True) if ds_smriprep is not None else None
    layout_output = BIDSLayout(path_output, validate=False, is_derivative=True)

    subjects = layout_tedana.get_subjects() if subject is None else [subject]

    for subject in subjects:
        # Run first level GLM
        stats_imgs, entities = run_first_level_glm(
            layout_tedana, layout_raw, layout_fmriprep, subject, layout_mask=layout_smriprep, 
            atlas=atlas, desc=desc, resample_mask=resample_mask, trial_type=trial_type, 
            modulator=modulator, interaction=interaction, modulator_split=modulator_split
        )

        # Run fixed or random effect model
        if model=='fixed':
            run_fixed_effect(stats_imgs, entities, layout_output, mask=atlas)
        elif model=='random':
            run_second_level_glm(stats_imgs, entities, layout_output)
        elif model=='both':
            run_fixed_effect(stats_imgs, entities, layout_output, mask=atlas)
            run_second_level_glm(stats_imgs, entities, layout_output)

    emissions = tracker.stop()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()