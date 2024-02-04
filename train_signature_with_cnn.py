import os 
import numpy as np
import cv2
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


print(">> Inisialiasi ----------------------------------")
root_folder = "datasets"
nm_class = ("Palsu","Asli")
lebar_gambar_resized = 50
tinggi_gambar_resized = 50
folderlists = os.listdir(root_folder) 
print("Ada %i Subjek yang memiliki masing-masing 6 tanda tangan Asli dan 6 Tanda Tangan Palsu" % len(folderlists))
print("Yaitu:")
for nama_subjek in folderlists:
    print(nama_subjek)
n_class = 2 #len(folderlists)
print("-----------------------------------------")
print("Jumlah Kelas sebanyak %i " % n_class)

datasets = []
labels = list()
y = []

print(">> EXTRAKSI FITUR DENGAN METODE BINERISASI (THRESHOLDING) ----------------------------------")
str_line_msg = "Mengekstraksi TTD {} - Kategori {} {}x"
for folder in folderlists:
    subFolders = os.listdir(root_folder + '/' + folder)
    for imgFolder in subFolders:
         files = os.listdir(root_folder + '/' + folder + "/" + imgFolder)
         nm_label = folder + "[" + imgFolder.upper() + "]"
         labels.append(nm_label)
         print (str_line_msg.format(folder,imgFolder.upper(),len(files)))
         for imageFile in files:           
            imagePath = root_folder + '/' + folder + "/" +  imgFolder + '/' + imageFile
            image = cv2.imread(imagePath) 
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
            #img_yuv[:,:,0] = cv2.equalizeHist(gray)
            (T, threshInv) = cv2.threshold( gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            image = cv2.resize(threshInv, (lebar_gambar_resized, tinggi_gambar_resized))       
            datasets.append(image)
            if imgFolder=='asli':
                y.append(1)
            else:
                y.append(0)
            
 


print("Jumlah Data Keseluruhan Sebanyak %i" % len(y))
print("------------------------------------------------------------------")
ukuran_data_uji = 0.2 # persentase data digunakan untuk data uji  
print("Mengacak dan Memilah Data Latih dan Data Uji, Persentase Data Uji Sebesar %d" % (ukuran_data_uji*100),"%")
train_X, test_X, train_Y, test_Y = train_test_split(datasets, y, test_size=ukuran_data_uji, random_state=1) 
print("Sehingga terdapat {} Data Latih dan {} Data Uji ".format(len(train_Y ),len(test_Y)))
#plt.show()

train_X = np.array(train_X)
test_X = np.array(test_X)
train_Y = np.array(train_Y)
test_Y = np.array(test_Y)
 
 
train_X = train_X.reshape(-1, lebar_gambar_resized,tinggi_gambar_resized, 1)
test_X = test_X.reshape(-1, lebar_gambar_resized,tinggi_gambar_resized, 1)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255
test_X = test_X / 255
 
# #print("Train Y:" ,train_Y)

train_Y_one_hot = to_categorical(train_Y,n_class)
test_Y_one_hot = to_categorical(test_Y,n_class)
 
print("------------------------------------------------------------------")
print("Rangkuman Model CNN")
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(50, 50, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Dense(n_class))
model.add(Activation('softmax'))
model.summary()

num_hidden_layers = len(model.layers) - 2   
print("Number of hidden layers:", num_hidden_layers)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

optimizer = model.optimizer
learning_rate = optimizer.learning_rate.numpy()
print("Learning Rate %f" % learning_rate)

print("------------------------------------------------------------------") 

print(">> MENJALANKAN PELATIHAN/PEMBELAJARAN ----------------------------------")
max_epoch = 3
print("Menjalankan Pelatihan Sebanyak %i" % max_epoch,"Iterasi")
train_result = model.fit(train_X, train_Y_one_hot, batch_size=1, epochs=max_epoch,validation_data=(test_X, test_Y_one_hot))
 
test_loss, test_acc = model.evaluate(test_X, test_Y_one_hot)
print(">> MENJALANKAN TESTING/EVALUASI ----------------------------------")
#print('Test loss', test_loss)
print('Akurasi Mencapai ', test_acc*100, " %")
print(">> MENJALANKAN PREDIKSI/VERIFIKASI ----------------------------------")
predictions = model.predict(test_X)

accuracy = train_result.history['accuracy'] 
loss = train_result.history['loss']
 
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, loss, 'b', label='loss')
plt.title('Accuracy and Loss')
plt.legend() 
plt.show()




# Create a figure and subplots
ncols = 8
nrows = 4
fig, axes = plt.subplots(nrows=nrows, ncols=ncols,)


print("Hasilnya:")
for idx,classPredicted in enumerate(predictions):
    test_class = test_Y[idx]
    predict_class = np.argmax(np.round(predictions[idx]))
    print("Data Uji {} dengan TTD {} Diprediksi {} [{}]".format(idx+1, nm_class[test_class] , nm_class[predict_class],"Benar" if test_class==predict_class else "Salah"))

    #print("Prediksi Hasil (Test Vs Actual): ", labels[np.argmax(np.round(predictions[idx]))], " vs ", labels[test_Y[idx]])
  
idxrow = 0
idxcol = 0

for idx,classPredicted in enumerate(predictions):
    test_class = test_Y[idx]
    predict_class = np.argmax(np.round(predictions[idx]))
    if idx >=(ncols*nrows):
        break
   
    axes[idxrow][idxcol].set_xticks([])
    # Remove y-axis ticks
    axes[idxrow][idxcol].set_yticks([])
    axes[idxrow][idxcol].imshow(test_X[idx], cmap='binary')
    axes[idxrow][idxcol].set_title(nm_class[predict_class] +"(" + nm_class[test_class] + ")")
    idxcol +=1
    
    if idxcol==ncols: 
        idxrow +=1
        idxcol = 0

            # plt.grid(False)  

totViews = ncols*nrows if (ncols*nrows) < len(test_Y) else len(test_Y)
fig.suptitle("Menampilkan Data Uji dan Hasil Prediksi(Aktual) Sebanyak {} dari {}".format(totViews,len(test_Y)))
plt.tight_layout() 
plt.show()