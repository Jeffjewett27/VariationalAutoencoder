import tensorflow as tf

# Image size for our model.
MODEL_INPUT_IMAGE_SIZE = [ 128 , 128 ]

# Fraction of the dataset to be used for testing.
TRAIN_TEST_SPLIT = 0.3

# Number of samples to take from dataset
NUM_SAMPLES = 20000

# Trick to one-hot encode the label.
y1 = tf.constant( [ 1. , 0. ] , dtype='float32' ) 
y2 = tf.constant( [ 0. , 1. ] , dtype='float32' ) 

# This method will be mapped for each filename in `list_ds`. 
def parse_image( filename ):

    # Read the image from the filename and resize it.
    image_raw = tf.io.read_file( filename )
    image = tf.image.decode_jpeg( image_raw , channels=3 ) 
    image = tf.image.resize( image , MODEL_INPUT_IMAGE_SIZE ) / 255

    # Split the filename to get the age and the gender. Convert the age ( str ) and the gender ( str ) to dtype float32.
    parts = tf.strings.split( tf.strings.split( filename , '/' )[ 4 ] , '_' )

    # One-hot encode the label
    gender = tf.strings.to_number( parts[ 1 ] )
    gender_onehot = ( gender * y2 ) + ( ( 1 - gender ) * y1 )

    return image , gender_onehot

# List all the image files in the given directory.
list_ds = tf.data.Dataset.list_files( r'C:\Users\Owner\Datasets\utkcropped\*' , shuffle=True )
# Map `parse_image` method to all filenames.
dataset = list_ds.map( parse_image , num_parallel_calls=tf.data.AUTOTUNE )
#dataset = dataset.take( NUM_SAMPLES )

# Create train and test splits of the dataset.
num_examples_in_test_ds = int( dataset.cardinality().numpy() * TRAIN_TEST_SPLIT )

test_ds = dataset.take( num_examples_in_test_ds )
train_ds = dataset.skip( num_examples_in_test_ds )

print( 'Num examples in train ds {}'.format( train_ds.cardinality() ) )
print( 'Num examples in test ds {}'.format( test_ds.cardinality() ) )