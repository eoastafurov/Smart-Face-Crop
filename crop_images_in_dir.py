import os
from cropper import Cropper
import cv2
import shutil


def copy_folders_structure_relative_path(images_dir, result_path):
    """
    Replicates folders structure from './images_dir/' to './result_path/images_dir/'
    Parameters:
        images_dir - directory name without symbols '.' or '/'
        result_path - relative path without symbol '.'
    """
    for path_from_top, _, files in os.walk(images_dir):
        new_path = '/'.join([result_path, images_dir] + path_from_top.split('/')[1:])
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            print('Making new dir: ./' + new_path)


def copy_folders_structure_absolute_path(
        entry: str,
        destination: str
):
    destination_concat = '/'.join([destination, entry.split('/')[-1:][0]])

    assert not os.path.exists(destination_concat)

    def list_files(directory, files):
        return [
            file for file in files if os.path.isfile(os.path.join(directory, file))
        ]

    shutil.copytree(
        src=entry,
        dst=destination_concat,
        ignore=list_files
    )

    return destination_concat


def crop_images_in_dir_abs_path(
        images_path: str,
        result_path: str,
        supported_files=('jpg', 'png', 'JPG', 'PNG', 'jpeg'),
        max_yaw=0.025,
        max_pitch=0.017,
        max_roll=0.34,
        HEIGHT=320,
        WIDTH=240,
        fill_color='white'
):
    """
    This function will copy folders structure from images_path to result_path
    and crops all images with supported type.
    Unhandled images will be placed in result_path/unhandled
    with two tags: unfiltered and without_landmarks.
    Parameters:
        images_path - absolute system path without symbols '.' and without ending '/'
        result_path - the same as absolute_path
    """
    cr = Cropper(
        max_yaw=max_yaw,
        max_pitch=max_pitch,
        max_roll=max_roll,
        HEIGHT=HEIGHT,
        WIDTH=WIDTH,
        fill_color=fill_color
    )
    assert not images_path.endswith('/')
    assert not result_path.endswith('/')
    assert not images_path.__contains__('.')
    assert not result_path.__contains__('.')

    destination_concat = copy_folders_structure_absolute_path(images_path, result_path)
    working_dir = images_path.split('/')[-1:][0]

    for path_from_top, subdirs, files in os.walk(images_path):
        for f in files:
            if not f.endswith(supported_files):
                continue

            image = cv2.imread(path_from_top + '/' + f)

            print("Working with " + f)
            res, _ = cr.process_image(image)

            if res is not None:
                save_path = destination_concat + path_from_top.split(working_dir)[1]
                cv2.imwrite(save_path + '/' + f, res)

    unfiltered_images = cr.unfiltered_images
    images_without_landmarks = cr.images_without_landmarks

    unfiltered_path = destination_concat + '/unhandled/' + 'unfiltered'
    without_landmarks_path = destination_concat + '/unhandled/' + 'without_landmarks'

    if not os.path.exists(unfiltered_path):
        os.makedirs(unfiltered_path)

    if not os.path.exists(without_landmarks_path):
        os.makedirs(without_landmarks_path)

    for i, img in enumerate(unfiltered_images):
        cv2.imwrite(unfiltered_path + '/' + str(i) + '.jpg', img)

    for i, img in enumerate(images_without_landmarks):
        cv2.imwrite(without_landmarks_path + '/' + str(i) + '.jpg', img)




def crop_images_dir_cascade(
        images_dir: str,
        result_dir: str,
        unhandled_images_dir='unhandled',
        supported_files=('jpg', 'png', 'JPG', 'PNG', 'jpeg'),
        max_yaw=0.025,
        max_pitch=0.017,
        max_roll=0.34,
        HEIGHT=320,
        WIDTH=240,
        fill_color='white'
):
    """
    NOTE: !you should use only path relative to working directory!

    This function will copy folders structure
    from './images_dir' to './result_dir/images_dir/
    and crops all images with supported type.
    Unhandled images will be placed in './unhandled/images_dir'
    with two tags: unfiltered and without_landmarks.
    Parameters:
        images_dir - dir under working directory without symbols '.' and without ending '/'
        result_dir - the same as absolute_path
    """
    cr = Cropper(
        max_yaw=max_yaw,
        max_pitch=max_pitch,
        max_roll=max_roll,
        HEIGHT=HEIGHT,
        WIDTH=WIDTH,
        fill_color=fill_color
    )

    assert not images_dir.__contains__('.')
    assert not result_dir.__contains__('.')

    copy_folders_structure_relative_path(images_dir, result_dir)

    for path_from_top, subdirs, files in os.walk(images_dir):
        for f in files:
            if not f.endswith(supported_files):
                continue

            image = cv2.imread(path_from_top + '/' + f)

            print("Working with " + f)
            res, _ = cr.process_image(image)

            if res is not None:
                save_path = '/'.join([result_dir, images_dir] + path_from_top.split('/')[1:])
                cv2.imwrite(save_path + '/' + f, res)

    unfiltered_images = cr.unfiltered_images
    images_without_landmarks = cr.images_without_landmarks

    unfiltered_path = './' + unhandled_images_dir + '/' + images_dir + '/' + 'unfiltered'
    without_landmarks_path = './' + unhandled_images_dir + '/' + images_dir + '/' + 'without_landmarks'

    if not os.path.exists(unfiltered_path):
        os.makedirs(unfiltered_path)

    if not os.path.exists(without_landmarks_path):
        os.makedirs(without_landmarks_path)

    for i, img in enumerate(unfiltered_images):
        cv2.imwrite(unfiltered_path + '/' + str(i) + '.jpg', img)

    for i, img in enumerate(images_without_landmarks):
        cv2.imwrite(without_landmarks_path + '/' + str(i) + '.jpg', img)


# crop_images_dir_cascade(
#     images_dir='ColortypesDone5',
#     result_dir='cropped',
#
#     # below are defaults
#     unhandled_images_dir='unhandled',
#     supported_files=('jpg', 'png', 'JPG', 'PNG', 'jpeg'),
#     max_yaw=0.025,
#     max_pitch=0.017,
#     max_roll=0.34,
#     HEIGHT=320,
#     WIDTH=240,
#     fill_color='white'
# )


crop_images_in_dir_abs_path(
    images_path='/Users/evgenijastafurov/Desktop/2021/Imagin/faceengine/mp_crop/ColortypesDone2',
    result_path='/Users/evgenijastafurov/Desktop/work/test_copy_destination',

    # below are defaults
    supported_files=('jpg', 'png', 'JPG', 'PNG', 'jpeg'),
    max_yaw=0.025,
    max_pitch=0.017,
    max_roll=0.34,
    HEIGHT=320,
    WIDTH=240,
    fill_color='white'
)
