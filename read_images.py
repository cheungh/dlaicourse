history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=20,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)

import numpy as np
#from google.colab import files
from keras.preprocessing import image

#uploaded = files.upload()

#for fn in uploaded.keys():
 
# predicting images
for fn in os.listdir("/tmp/contents"):

    path = '/tmp/contents/' + fn
    img = image.load_img(path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0]>0.5:
        print(fn + " is a human")
    else:
        print(fn + " is a horse")
