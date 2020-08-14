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

# All classes
classes_all = {
    'A001': 'drink_water',
    'A002': 'eat_meal',
    'A003': 'brush_teeth',
    'A004': 'brush_hair',
    'A005': 'drop',
    'A006': 'pick_up',
    'A007': 'throw',
    'A008': 'sit_down',
    'A009': 'stand_up',
    'A010': 'clapping',
    'A011': 'reading',
    'A012': 'writing',
    'A013': 'tear_up_paper',
    'A014': 'put_on_jacket',
    'A015': 'take_off_jacket',
    'A016': 'put_on_a_shoe',
    'A017': 'take_off_a_shoe',
    'A018': 'put_on_glasses',
    'A019': 'take_off_glasses',
    'A020': 'put_on_a_hat_or_cap',
    'A021': 'take_off_a_hat_or_cap',
    'A022': 'cheer_up',
    'A023': 'hand_waving',
    'A024': 'kicking_something',
    'A025': 'reach_into_pocket',
    'A026': 'hopping',
    'A027': 'jump_up',
    'A028': 'phone_call',
    'A029': 'play_with_phone_or_tablet',
    'A030': 'type_on_a_keyboard',
    'A031': 'point_to_something',
    'A032': 'taking_a_selfie',
    'A033': 'check_time',
    'A034': 'rub_two_hands',
    'A035': 'nod_head_or_bow',
    'A036': 'shake_head',
    'A037': 'wipe_face',
    'A038': 'salute',
    'A039': 'put_palms_together',
    'A040': 'cross_hands_in_front',
    'A041': 'sneeze',
    'A042': 'staggering',
    'A043': 'falling_down',
    'A044': 'headache',
    'A045': 'chest_pain',
    'A046': 'back_pain',
    'A047': 'neck_pain',
    'A048': 'nausea',
    'A049': 'fan_self',
    'A050': 'punch_or_slap',
    'A051': 'kicking',
    'A052': 'pushing',
    'A053': 'pat_on_back',
    'A054': 'point_finger',
    'A055': 'hugging',
    'A056': 'giving_object',
    'A057': 'touch_pocket',
    'A058': 'shaking_hands',
    'A059': 'walking_towards',
    'A060': 'walking_apart',
    'A061': 'put_on_headphone',
    'A062': 'take_off_headphone',
    'A063': 'shoot_a_basket',
    'A064': 'bounce_ball',
    'A065': 'tennis_bat_swing',
    'A066': 'juggle_table_tennis_ball',
    'A067': 'hush',
    'A068': 'flick_hair',
    'A069': 'thumb_up',
    'A070': 'thumb_down',
    'A071': 'make_ok_sign',
    'A072': 'make_victory_sign',
    'A073': 'staple_book',
    'A074': 'counting_money',
    'A075': 'cutting_nails',
    'A076': 'cutting_paper',
    'A077': 'snap_fingers',
    'A078': 'open_bottle',
    'A079': 'sniff_or_smell',
    'A080': 'squat_down',
    'A081': 'toss_a_coin',
    'A082': 'fold_paper',
    'A083': 'ball_up_paper',
    'A084': 'play_magic_cube',
    'A085': 'apply_cream_on_face',
    'A086': 'apply_cream_on_hand',
    'A087': 'put_on_bag',
    'A088': 'take_off_bag',
    'A089': 'put_object_into_bag',
    'A090': 'take_object_out_of_bag',
    'A091': 'open_a_box',
    'A092': 'move_heavy_object',
    'A093': 'shake_fist',
    'A094': 'throw_up_cap_or_hat',
    'A095': 'capitulate',
    'A096': 'cross_arms',
    'A097': 'arm_circles',
    'A098': 'arm_swings',
    'A099': 'run_on_the_spots',
    'A100': 'butt_kicks',
    'A101': 'cross_toe_touch',
    'A102': 'side_kick',
    'A103': 'yawn',
    'A104': 'stretch_oneself',
    'A105': 'blow_nose',
    'A106': 'hit_with_object',
    'A107': 'wield_knife',
    'A108': 'knock_over',
    'A109': 'grab_stuff',
    'A110': 'shoot_with_gun',
    'A111': 'step_on_foot',
    'A112': 'high_five',
    'A113': 'cheers_and_drink',
    'A114': 'carry_object',
    'A115': 'take_a_photo',
    'A116': 'follow',
    'A117': 'whisper',
    'A118': 'exchange_things',
    'A119': 'support_somebody',
    'A120': 'rock-paper-scissors'
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


def get_cs_split(video_name):
    '''
    Returns if a video belongs to the train, validation os test dataset given its name by the cross-subject method

    :param video_name: video file name
    :return: 'train', 'val' or 'test'
    '''
    subject = video_name[0:4]

    if subject in train_subjects:
        return 'train'
    elif subject in val_subjects:
        return 'val'
    else:
        return 'test'


def get_cv_split(video_name):
    '''
    Returns if a video belongs to the train, validation os test dataset given its name by the cross-view method

    :param video_name: video file name
    :return: 'train', 'val' or 'test'
    '''
    camera = video_name[4:8]

    if camera in train_cameras:
        return 'train'
    elif camera in val_cameras:
        return 'val'
    else:
        return 'test'
