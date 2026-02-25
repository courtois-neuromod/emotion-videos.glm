# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from bids import BIDSLayout
from nilearn.image import resample_to_img
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel


def check_bids_input(bids_input, input_type, match):
    if len(bids_input) == 0:
        raise FileNotFoundError(f"No file {input_type} associated with {match}")
    elif len(bids_input) > 1:
        raise ValueError(f"More than one {input_type} file associated with {match}")
    else:
        return bids_input[0]


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


def get_mask(layout_smriprep, entities, file, resample_mask=True):
    # Retrieve file
    mask = layout_smriprep.get(subject=entities['subject'], space='MNI152NLin2009cAsym', desc='brain', suffix='mask', extension='.nii.gz')

   # Validate input
    mask = check_bids_input(mask, 'MNI152Lin2009cAsym mask', file.filename)

    # Resample mask to save
    if resample_mask:
        mask = resample_to_img(mask.path, file.path, interpolation='nearest')

    return mask


def run_first_level_glm(layout_tedana, layout_raw, layout_fmriprep, layout_smriprep, path_output, subject=None, desc='denoised', resample_mask=True, trial_type=None):
    
    logger = logging.getLogger(__name__)
    
    # Get subjects ids
    if subject is None:
        subjects = layout_tedana.get_subjects()
    else:
        subjects = [subject]

    sub_stats_imgs = {}

    for subject in subjects:
        stats_imgs = {}
        
        # Retrieve bold data for given `desc`
        files = layout_tedana.get(subject=subject, desc=desc, space='MNI152NLin2009cAsym', suffix='bold', extension='.nii.gz')
        for file in files:
            logger.info(f"... running GLM for: {file.filename}")

            entities = file.get_entities()
            # Retrieve repetition time
            r_t = get_tr(layout_fmriprep, entities, file.filename)
        
            # Retrieve events file
            events = get_events(layout_raw, entities, file.filename)

            if trial_type == 'Gif':
                exp_regressors = events[trial_type].str.replace('.mp4', '', regex=False)
            else:
                exp_regressors = ['StimvIti']*len(events)
            
            events = pd.DataFrame({'trial_type': exp_regressors, 'onset': events['onset_video_flip'], 'duration': events['total_duration']})

            # Retrieve mask
            mask = get_mask(layout_smriprep, entities, file, resample_mask=resample_mask)

            logger.info(f"... fitting the First level model with the following parameters: ")
            logger.info(f"...     mask: {mask}")
            logger.info(f"...     bold: {file.path}")
            logger.info(f"...     rt: {r_t}")
            logger.info(f"...     events: {events}")

            # First level GLM
            first_level_model = FirstLevelModel(r_t, mask_img=mask, n_jobs=-1)
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

        if trial_type == 'Gif':
            sub_stats_imgs.update({
                subject: stats_imgs
            })
        else:
            logger.info(f"... computing second level contrast")

            betas = [simg['effect_size'] for simg in stats_imgs['StimvIti']]

            second_level_contrast = 'intercept'
            second_level_model = SecondLevelModel(n_jobs=-1)
            second_level_model = second_level_model.fit(betas, design_matrix=pd.DataFrame({second_level_contrast: [1]*len(betas)}))
            
            zmap = second_level_model.compute_contrast(
                second_level_contrast=second_level_contrast, second_level_stat_type='t', output_type='z_score'
            )

            # Save output
            out_name = Path(
                path_output,
                f"sub-{subject}",
                f"{subject}_task-emotion_space-MNI152NLin2009cAsym_contrast-{second_level_contrast}_stat-z_statmap.nii.gz"
            )
            Path(out_name).parent.mkdir(parents=True, exist_ok=True)
            zmap.to_filename(out_name)

    if trial_type == 'Gif':
        return sub_stats_imgs


@click.command()
@click.argument('ds_tedana', type=str)
@click.argument('ds_raw', type=click.Path())
@click.argument('ds_fmriprep', type=click.Path())
@click.argument('ds_smriprep', type=click.Path())
@click.argument('path_output', type=click.Path())
@click.option('--subject', type=str, default=None, help='')
@click.option('--desc', type=str, default='denoised', help='')
@click.option('--trial_type', type=str, default=None, help='')
@click.option('--resample_mask', is_flag=True, help='Flag to specify if XXX')
def main(ds_tedana, ds_raw, ds_fmriprep, ds_smriprep, path_output, subject, desc, trial_type, resample_mask):

    # Defined BIDSLayouts
    layout_raw = BIDSLayout(ds_raw, validate=False, is_derivative=True)
    layout_smriprep = BIDSLayout(ds_smriprep, validate=False, is_derivative=True)
    layout_fmriprep = BIDSLayout(ds_fmriprep, validate=False, is_derivative=True)
    layout_tedana = BIDSLayout(ds_tedana, validate=False, is_derivative=True)


    run_first_level_glm(layout_tedana, layout_raw, layout_fmriprep, layout_smriprep, path_output, subject=subject, desc=desc, resample_mask=resample_mask, trial_type=trial_type)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()