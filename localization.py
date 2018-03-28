import ujson as json

anno_classes = ['alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft']

bb_json = {}
for c in anno_classes:

    if c == 'other': continue # no annotation file for "other" class

    j = json.load(open('{}annos/{}_labels.json'.format(path, c), 'r'))

    for l in j:
        if 'annotations' in l.keys() and len(l['annotations'])>0:
            bb_json[l['filename'].split('/')[-1]] = sorted(
                l['annotations'], key=lambda x: x['height']*x['width'])[-1]




bb_json['img_04908.jpg']


file2idx = {o:i for i,o in enumerate(raw_filenames)}
val_file2idx = {o:i for i,o in enumerate(raw_val_filenames)}


empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}


for f in raw_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox


for f in raw_val_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox



bb_params = ['height', 'width', 'x', 'y']

def convert_bb(bb, size):
    bb = [bb[p] for p in bb_params]
    conv_x = (224. / size[0])
    conv_y = (224. / size[1])
    bb[0] = bb[0]*conv_y
    bb[1] = bb[1]*conv_x
    bb[2] = max(bb[2]*conv_x, 0)
    bb[3] = max(bb[3]*conv_y, 0)
    return bb


trn_bbox = np.stack([convert_bb(bb_json[f], s) for f,s in zip(raw_filenames, sizes)], 
                   ).astype(np.float32)
val_bbox = np.stack([convert_bb(bb_json[f], s) 
                   for f,s in zip(raw_val_filenames, raw_val_sizes)]).astype(np.float32)



inp = Input(shape=(224, 224, 3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))
x = Conv2D(64, (3, 3), activation='relu', padding='same')(inp)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x_bb = Dense(4, name='bb')(x)
x_class = Dense(8, activation='softmax', name='class')(x)


model = Model([inp], [x_bb, x_class])
model.compile(Adam(lr=0.001), loss=['mse', 'categorical_crossentropy'], metrics=['accuracy'],
             loss_weights=[.001, 1.])

model.fit(X_train, [trn_bbox, Y_train], batch_size=batch_size, nb_epoch=3, 
             validation_data=(X_valid, [val_bbox, Y_valid]))








