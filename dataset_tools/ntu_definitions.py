import os

# 25 classes
classes = {
    'A001': 'drink_water',
    'A002': 'eat_meal',
    'A003': 'brush_teeth',
    'A008': 'sit_down',
    'A009': 'stand_up',
    'A011': 'reading',
    'A012': 'writing',
    'A027': 'jump_up',
    'A028': 'phone_call',
    'A032': 'taking_a_selfie',
    'A037': 'wipe_face',
    'A041': 'sneeze',
    'A043': 'falling_down',
    'A044': 'headache',
    'A045': 'chest_pain',
    'A046': 'back_pain',
    'A047': 'neck_pain',
    'A048': 'nausea',
    'A069': 'thumb_up',
    'A070': 'thumb_down',
    'A074': 'counting_money',
    'A085': 'apply_cream_on_face',
    'A103': 'yawn',
    'A104': 'stretch_oneself',
    'A105': 'blow_nose'
}

# 106 subjects total

# 53 training subjects
train_subjects = ['S001', 'S002', 'S004', 'S005', 'S008', 'S009', 'S013', 'S014', 'S015', 'S016', 'S017', 'S018',
                  'S019', 'S025', 'S027', 'S028', 'S031', 'S034', 'S035', 'S038', 'S045', 'S046', 'S047', 'S049',
                  'S050', 'S052', 'S053', 'S054', 'S055', 'S056', 'S057', 'S058', 'S059', 'S070', 'S074', 'S078',
                  'S080', 'S081', 'S082', 'S083', 'S084', 'S085', 'S086', 'S089', 'S091', 'S092', 'S093', 'S094',
                  'S095', 'S097', 'S098', 'S100', 'S103']

# 27 validation subjects
val_subjects = ['S006', 'S010', 'S012', 'S020', 'S022', 'S024', 'S026', 'S030', 'S032', 'S036', 'S040', 'S042',
                'S044', 'S048', 'S060', 'S062', 'S064', 'S066', 'S068', 'S072', 'S076', 'S088', 'S090', 'S096',
                'S102', 'S104', 'S106']

# 26 test subjects
test_subjects = ['S003', 'S007', 'S011', 'S021', 'S023', 'S029', 'S033', 'S037', 'S039', 'S041', 'S043', 'S051',
                 'S061', 'S063', 'S065', 'S067', 'S069', 'S071', 'S073', 'S075', 'S077', 'S079', 'S087', 'S099',
                 'S101', 'S105']

train_cameras = ['C001']
val_cameras = ['C002']
test_cameras = ['C003']


def get_cs_split(files):
    '''
    Returns train, validation and test cross-subject splits

    :param files: Array with every file path. The files must be named in the default NTU format
    :return: Train, validation and test cross-subject splits
    '''
    train = []
    validation = []
    test = []

    for file in files:
        path, file_name = os.path.split(file)
        subject = file_name[0:4]
        if subject in train_subjects:
            train.append(file)
        elif subject in val_subjects:
            validation.append(file)
        else:
            test.append(file)

    return train, validation, test


def get_cv_split(files):
    '''
    Returns train, validation and test cross-view splits

    :param files: Array with every file path. The files must be named in the default NTU format
    :return: Train, validation and test cross-view splits
    '''
    train = []
    validation = []
    test = []

    for file in files:
        path, file_name = os.path.split(file)
        camera = file_name[4:8]
        if camera in train_cameras:
            train.append(file)
        elif camera in val_cameras:
            validation.append(file)
        else:
            test.append(file)

    return train, validation, test
