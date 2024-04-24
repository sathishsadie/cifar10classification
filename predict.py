from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
model=load_model('cifar.h5')
classes = [
    "airplane",
    "automobile",  
    "bird", 
    "cat",  
    "deer",  
    "dog",  
    "frog",  
    "horse",  
    "ship",  
    "truck" 
]
iii=image.load_img('cat.png')
iamg=image.load_img('cat.png',target_size=(32,32,3))
img=image.img_to_array(iamg)
img=np.expand_dims(img,axis=0)
predi=model.predict(img)
cl=classes[np.argmax(predi)]
plt.imshow(iii)
plt.axis('off')
plt.title(cl)
plt.show()