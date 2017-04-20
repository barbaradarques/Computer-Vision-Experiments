from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

model = VGG16(weights='imagenet', include_top=False) # top = camadas fully-connected

# all_weights = []
# for layer in model.layers:
#    w = layer.get_weights()
#    all_weights.append(w)
# print(all_weights)

img_path = 'elephant.jpg' # imagem teste
img = image.load_img(img_path, target_size=(224, 224)) # carrega imagem e a redimensiona
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) # adiciona um eixo extra
x = preprocess_input(x)

features = model.predict(x)
# print('Predicted:', decode_predictions(features, top=3)[0]) # para classificar deve-se ativar as camadas FC


## Tentativa de imprimir a saída das camadas convolucionais como imagem ##
# out = np.squeeze(features, axis=0) # remove o eixo adicionado à mão
# print(out.shape)  # Dimensão: (7,7, 512)
# out2 = K.permute_dimensions(out, (2, 0, 1))
# print(out2.shape)  # Dimensão: (512, 7, 7) -> movi os eixos para ver se eu conseguia imprimir (sem sucesso)
# plt.imshow(out2[0])

## Tentativa de imprimir a saída da terceira camada convolucional como imagem ##
get_feature = K.function([model.layers[0].input, 
              K.learning_phase()], [model.layers[3].output])
feat = get_feature([x, 0])[0]
feat2 = np.squeeze(feat, axis=0) # remove eixo extra
print(feat2.shape) # Dimensão: (112, 112, 64)
plt.imshow(feat2)