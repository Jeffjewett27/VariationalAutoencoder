import tensorflow as tf
from tensorflow._api.v2 import data
import tensorflow_datasets as tfds

# Image size for our model.
MODEL_INPUT_IMAGE_SIZE = [ 128 , 128 ]

# Fraction of the dataset to be used for testing.
TRAIN_TEST_SPLIT = 0.3

# Number of samples to take from dataset
NUM_SAMPLES = 20000

# This method will be mapped for each filename in `list_ds`. 
def parse_image( filename ):

    # Read the image from the filename and resize it.
    image_raw = tf.io.read_file( filename )
    image = tf.image.decode_jpeg( image_raw , channels=3 ) 
    image = tf.image.resize( image , MODEL_INPUT_IMAGE_SIZE ) / 255

    # Split the filename to get the age and the gender. Convert the age ( str ) and the gender ( str ) to dtype float32.
    parts = tf.strings.split( tf.strings.split( filename , '/' )[ 4 ] , '_' )

    # One-hot encode the label
    age = tf.strings.to_number( parts[0] ) / 100
    gender = tf.strings.to_number( parts[ 1 ] )
    race = tf.one_hot(tf.strings.to_number(parts[2], out_type=tf.dtypes.int32), 5)

    #return image , age, gender, race
    return image

def split_datasets(dataset):
    tensors = {}
    names = ["image", "age", "gender", "race"]
    for name in names:
        tensors[name] = tfds.as_numpy(dataset.map(lambda x: x[name]))

    return tensors

def get_face_data():
    # List all the image files in the given directory.
    list_ds = tf.data.Dataset.list_files( r'/home/CS/users/jjewett/.linux/datasets/utkcropped/*.jpg' , shuffle=True )
    # Map `parse_image` method to all filenames.
    dataset = list_ds.map( parse_image , num_parallel_calls=tf.data.AUTOTUNE )
    #d1 = dataset.map(lambda a, b: a)
    #d2 = dataset.map(lambda a, b: b)
    #print(d2)
    #splitdataset = split_datasets(dataset)
    #print(dataset)
    # Create train and test splits of the dataset.
    #num_examples_in_test_ds = int( dataset.cardinality().numpy() * TRAIN_TEST_SPLIT )

    #test_ds = dataset.take( num_examples_in_test_ds )
    #train_ds = dataset.skip( num_examples_in_test_ds )
    #test_ds = test_ds.shuffle(buffer_size=num_examples_in_test_ds)
    #train_ds = train_ds.shuffle(buffer_size=int(dataset.cardinality().numpy()))
    #print(train_ds.cardinality().numpy())
    #return train_ds.batch(8), test_ds.batch(8)
    return dataset

if __name__ == "__main__":
    data = get_face_data()
    
    #iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
    #iterator = iter(data)
    #data_X, data_y = iterator.get_next()
    #print(data_X, data_y)
    for i, j in data:
        print(i)
