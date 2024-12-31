# DenseNet201

base_model = tf.keras.applications.densenet.DenseNet201(weights="imagenet",include_top=False,input_shape =(width, height,channel))

for layer in base_model.layers:
    layer.trainable = False

# Create a new model and add the NasNetLarge base model
Dense201_model = Sequential()
Dense201_model.add(base_model)
Dense201_model.add(GlobalAveragePooling2D())
Dense201_model.add(Dense(256, activation='relu'))
Dense201_model.add(Dense(num_classes, activation='softmax'))

opt = tf.keras.optimizers.SGD(learning_rate=0.001)

Dense201_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

Dense201_hist = Dense201_model.fit(train_im, train_lb, epochs=my_epochs, batch_size=my_batch_size, validation_split=0.2)
#### 
Dense201_model.evaluate(train_im, train_lb)

####

y_pred_Dense201=Dense201_model.predict(test_im,verbose=0)
y_pred_Dense201=np.argmax(y_pred_Dense201, axis=1)

y_test = [np.argmax(element) for element in test_lb]
accuracy_score(y_test, y_pred_Dense201)

classfi_report=classification_report(y_test, y_pred_Dense201,output_dict=True)

K=0
accuracy_array[K] =accuracy_score(y_test, y_pred_Dense201)
precision_array[K] = classfi_report['macro avg']['precision']
recall_array[K] = classfi_report['macro avg']['recall']
f1_score_array[K] = classfi_report['macro avg']['f1-score']
print("accuracy : %.3f" % accuracy_array[K])
print("precision: %.3f" % precision_array[K])
print("recall: %.3f" % recall_array[K])
print("f1 score: %.3f" % f1_score_array[K])

####

Con_matrix=confusion_matrix(y_test, y_pred_Dense201)
fig, ax = plot_confusion_matrix(conf_mat=Con_matrix,
                            show_absolute=True,
                            show_normed=False,
                            colorbar=True)

ax.set_title('confusion_matrix of Dense201 ')
plt.show()

####

# training and validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(Dense201_hist.history['accuracy'], label='Training Accuracy')
plt.plot(Dense201_hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy Dense201')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# training and validation loss
plt.subplot(1, 2, 2)
plt.plot(Dense201_hist.history['loss'], label='Training Loss')
plt.plot(Dense201_hist.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Dense201')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


## VGG16

base_model = tf.keras.applications.vgg16.VGG16(weights="imagenet",include_top=False,input_shape =(width, height,channel))

for layer in base_model.layers:
    layer.trainable = False

# Create a new model and add the NasNetLarge base model
vgg16_model = Sequential()
vgg16_model.add(base_model)
vgg16_model.add(GlobalAveragePooling2D())
vgg16_model.add(Dense(256, activation='relu'))

vgg16_model.add(Dense(num_classes, activation='softmax'))

opt = tf.keras.optimizers.SGD(learning_rate=0.001)
vgg16_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

vgg16_hist = vgg16_model.fit(train_im, train_lb, epochs=my_epochs, batch_size=my_batch_size, validation_split=0.2)

####
vgg16_model.evaluate(test_im, test_lb)
####
y_pred_VGG16=vgg16_model.predict(test_im,verbose=0)
y_pred_VGG16=np.argmax(y_pred_VGG16, axis=1)

y_test = [np.argmax(element) for element in test_lb]

accuracy_score(y_test, y_pred_VGG16)

K=1
classfi_report=classification_report(y_test, y_pred_VGG16,output_dict=True)

accuracy_array[K] =accuracy_score(y_test, y_pred_VGG16)
precision_array[K] = classfi_report['macro avg']['precision']
recall_array[K] = classfi_report['macro avg']['recall']
f1_score_array[K] = classfi_report['macro avg']['f1-score']
print("\accuracy : %.3f" % accuracy_array[K])
print("precision: %.3f" % precision_array[K])
print("recall: %.3f" % recall_array[K])
print("f1 score: %.3f" % f1_score_array[K])

####

Con_matrix=confusion_matrix(y_test, y_pred_VGG16)
fig, ax = plot_confusion_matrix(conf_mat=Con_matrix,
                            show_absolute=True,
                            show_normed=False,
                            colorbar=True)

ax.set_title('confusion_matrix of VGG16')
plt.show()

####

# training and validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(vgg16_hist.history['accuracy'], label='Training Accuracy')
plt.plot(vgg16_hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy VGG16')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# training and validation loss
plt.subplot(1, 2, 2)
plt.plot(vgg16_hist.history['loss'], label='Training Loss')
plt.plot(vgg16_hist.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss VGG16')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

## ResNet101

base_model = tf.keras.applications.resnet_v2.ResNet101V2(weights="imagenet",include_top=False,input_shape =(width, height,channel))

for layer in base_model.layers:
    layer.trainable = False

# Create a new model and add the NasNetLarge base model
ResNet101V2_model = Sequential()
ResNet101V2_model.add(base_model)
ResNet101V2_model.add(GlobalAveragePooling2D())
ResNet101V2_model.add(Dense(256, activation='relu'))

ResNet101V2_model.add(Dense(num_classes, activation='softmax'))

opt = tf.keras.optimizers.SGD(learning_rate=0.001)

ResNet101V2_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
ResNet101V2_hist = ResNet101V2_model.fit(train_im, train_lb, epochs=my_epochs, batch_size=my_batch_size, validation_split=0.2)
###
ResNet101V2_model.evaluate(test_im, test_lb)
####
y_pred_ResNet101V2 = ResNet101V2_model.predict(test_im,verbose=0)
y_pred_ResNet101V2 = np.argmax(y_pred_ResNet101V2, axis=1)

y_test = [np.argmax(element) for element in test_lb]
accuracy_score(y_test, y_pred_ResNet101V2)


classfi_report=classification_report(y_test, y_pred_ResNet101V2,output_dict=True)

K=2

accuracy_array[K] =accuracy_score(y_test, y_pred_ResNet101V2)
precision_array[K] = classfi_report['macro avg']['precision']
recall_array[K] = classfi_report['macro avg']['recall']
f1_score_array[K] = classfi_report['macro avg']['f1-score']
print("accuracy : %.3f" % accuracy_array[K])
print("precision: %.3f" % precision_array[K])
print("recall: %.3f" % recall_array[K])
print("f1 score: %.3f" % f1_score_array[K])

####
Con_matrix=confusion_matrix(y_test, y_pred_ResNet101V2)
fig, ax = plot_confusion_matrix(conf_mat=Con_matrix,
                            show_absolute=True,
                            show_normed=False,
                            colorbar=True)
ax.set_title('confusion_matrix of ResNet101V2')
plt.show()

####

# training and validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(ResNet101V2_hist.history['accuracy'], label='Training Accuracy')
plt.plot(ResNet101V2_hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy ResNet101 v2')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# training and validation loss
plt.subplot(1, 2, 2)
plt.plot(ResNet101V2_hist.history['loss'], label='Training Loss')
plt.plot(ResNet101V2_hist.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss ResNet101 v2')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

## **Comparing the results of models**

#@title Default title text

H=6
L=4

print('--------------------result--------------------------')
fig1=plt.figure(figsize=(H, L)) #
plt.bar(algorithms_name, accuracy_array,color = ['royalblue', 'limegreen'])
plt.xticks(algorithms_name, rotation=90)
plt.ylabel('percent%')
plt.title('Accuracy of Model')
plt.xlabel("Algoritm names")
for i, v in enumerate(accuracy_array):
    v=round(v,3)
    plt.text(i-0.2 , v+0.01 , str(v), color='blue', fontweight='bold')
fig1.show()


fig2=plt.figure(figsize=(H, L)) #
plt.bar(algorithms_name, precision_array,color = ['royalblue', 'limegreen'])
plt.xticks(algorithms_name, rotation=90)
plt.ylabel('percent%')
plt.title('Precision of Models')
plt.xlabel("Algoritm names")
for i, v in enumerate(precision_array):
    v=round(v,3)
    plt.text(i-0.2 , v+0.01 , str(v), color='blue', fontweight='bold')
fig2.show()




fig3=plt.figure(figsize=(H, L)) #
plt.bar(algorithms_name, recall_array,color = ['royalblue', 'limegreen'])
plt.xticks(algorithms_name, rotation=90)
plt.ylabel('percent%')
plt.title('Recall of Models')
plt.xlabel("Algoritm names")
for i, v in enumerate(recall_array):
    v=round(v,3)
    plt.text(i-0.2 , v+0.01 , str(v), color='blue', fontweight='bold')
fig3.show()



fig4=plt.figure(figsize=(H, L)) #
plt.bar(algorithms_name, f1_score_array,color = ['royalblue', 'limegreen'])
plt.xticks(algorithms_name, rotation=90)
plt.ylabel('percent%')
plt.title('f1-score of Models')
plt.xlabel("Algoritm names")
for i, v in enumerate(f1_score_array):
    v=round(v,3)
    plt.text(i-0.2 , v+0.01 , str(v), color='blue', fontweight='bold')
fig4.show()

### **Interface**
np.array([test_im[0]]).shape

####

indices = [0, 20, 30, 50, 100, 55, 1, 18, 33, 46, 170, 57, 5, 25, 35, 40, 200, 10]
reverse_labs = {v: k for k, v in labs.items()}


fig, axs = plt.subplots(3, 6, figsize=(18, 10))

for i, ax in enumerate(axs.flat):
    index = indices[i]
    ax.imshow(test_im[index])

    y_pred = ResNet101V2_model.predict(np.array([test_im[index]]), verbose=0)
    y_pred = np.argmax(y_pred)


    GT = reverse_labs.get(np.argmax(test_lb[index]))
    pred = reverse_labs.get(y_pred)
    if GT == pred:
        c = 'green'
    else:
        c = 'red'
    ax.set_title(f"GT: {GT} - Pred: {pred}", color=c)
    ax.set_axis_off()

plt.subplots_adjust(wspace=0.3, hspace=0)
plt.show()
