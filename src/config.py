from modurec.features.feature import FeatureData

# Directories
data_dir = 'data'

# Compose stage
dataset = 'lab_data'
compose_target = 'data.pkl'

# Prepare stage
prepare_target = 'data_prepared.pkl'
window_size = 700
window_step = 100
random_seed = 42
augment_percent = 100
augment_params = {'sigma': 5}
augmenter = None
partial_windows = False
subsample = False

# Diagrams stage
diagrams_target = 'diagrams.pkl'
restrict = 10
columns = ['acc_x', 'acc_y']
to_calculate = [FeatureData(name='diagram', params={'dim': 2}),
                FeatureData(name='diagram', params={'dim': 3}),
                FeatureData(name='diagram', params={'dim': 4}),
                FeatureData(name='diagram', params={'dim': 10}),
                FeatureData(name='diagram', params={'dim': 2, 'kind': 'abs'}),
                FeatureData(name='diagram', params={'dim': 10, 'kind': 'abs'}),
                FeatureData(name='diagram', params={'dim': 2, 'kind': 'phi'}),
                FeatureData(name='diagram', params={'dim': 10, 'kind': 'phi'}),
                FeatureData(name='diagram', params={'dim': 2, 'kind': 'abs', 'fil': 'star'}),
                FeatureData(name='diagram', params={'dim': 2, 'kind': 'phi', 'fil': 'star'}),
                FeatureData(name='diagram', params={'dim': 2, 'step': 30}),
                FeatureData(name='diagram', params={'dim': 4, 'step': 30})]

# Feature stage
features_target = 'features.pkl'


# Dataset metadata
label_column = 'acc_gyro_event'
timestamp_column = 'acc_gyro_timestamp'
correct_labels = [0, 1, 2, 4, 8]
timestamp_multiplier = 1e-4
signal_columns = [['acc_x', 'acc_y', 'acc_z'],
                  ['gyro_x', 'gyro_y', 'gyro_z']]
scalar_columns = ['patient_id', 'measure_id', 'augmented']
datetime_column = 'packet_start_time'
patient_column = 'patient_id'
measure_column = 'measure_id'
augment_column = 'augmented'
