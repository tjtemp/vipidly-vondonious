def styler_transfer(
        STYLE_PATH=None,
        CONTENT_PATH=None):
    # this is remote testing but freaking slow
    import scipy.io
    import numpy as np
    import os
    import scipy.misc
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import time

    start = time.time()
    cwd = os.path.dirname(os.path.abspath(__file__))
    if STYLE_PATH == None:
        STYLE_PATH = cwd + '/../dataset/imgsample/pop_art_sample.jpg'
    if CONTENT_PATH == None:
        CONTENT_PATH = cwd + '/../dataset/imgsample/ted.jpg'

    DATASET_DIR = cwd + '/../dataset/'
    VGG19_PATH = DATASET_DIR + 'pretrained_model/imagenet-vgg-very-deep-19.mat'
    CONTENT_LAYER = 'relu2_2'
    STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')


    def VGG19(data_path, input_image):
        layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )
        data = scipy.io.loadmat(data_path)
        mean = data['normalization'][0][0][0]
        mean_pixel = np.mean(mean, axis=(0, 1))
        weights = data['layers'][0]
        net = {}
        current = input_image
        for i, name in enumerate(layers):
            kind = name[:4]
            if kind == 'conv':
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                current = _conv_layer(current, kernels, bias)
            elif kind == 'relu':
                current = tf.nn.relu(current)
            elif kind == 'pool':
                current = _pool_layer(current)
            net[name] = current
        assert len(net) == len(layers)
        return net, mean_pixel, layers

    def _conv_layer(input_, weights, bias):
        conv = tf.nn.conv2d(input_, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
        return tf.nn.bias_add(conv, bias)

    def _pool_layer(input_):
        return tf.nn.max_pool(input_, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

    def preprocess(image, mean_pixel):
        return image - mean_pixel

    def unprocess(image, mean_pixel):
        return image + mean_pixel

    def imread(path):
        return scipy.misc.imread(path).astype(np.float)

    def imsave(path, img):
        img = np.clip(img, 0, 255).astype(np.unit8)
        scipy.misc.imsave(path, img)


    raw_content = scipy.misc.imread(CONTENT_PATH)
#   plt.figure(0, figsize=(10, 5))
#   plt.imshow(raw_content)
#   plt.title('Original content image')
#   plt.show()
#

    cr = ''
    pr = ''
    delim = '<br/>'
    import mpld3

    # Extract Content image
    content_image = raw_content.astype(np.float)
    content_shape = (1,) + content_image.shape # newaxis
    with tf.Graph().as_default(), tf.Session() as sess, tf.device('/gpu:0'):
        image = tf.placeholder('float', shape=content_shape)
        nets, content_mean_pixel, _ = VGG19(VGG19_PATH, image)
        content_image_pre = np.array([preprocess(content_image, content_mean_pixel)])
        content_features = nets[CONTENT_LAYER].eval(feed_dict={image: content_image_pre})
        #print (" Type of 'features' is ", type(content_features))
        #print (" Shape of 'features' is %s" % (content_features.shape,))
        
        cr +=  " Type of 'features' is " + str(type(content_features)) + delim
        cr += " Shape of 'features' is %s" % (content_features.shape,) + delim
    
        #for i in range(5):
        #   plt.figure(i, figsize=(10, 5))
        #   plt.matshow(content_features[0, :, :, i], cmap=plt.cm.gray, fignum=i)
        #   plt.title("%d-layer content feature" % (i))
        #   plt.colorbar()
        #   plt.show()
            
    # Extract Style gram matrix
    raw_style    = scipy.misc.imread(STYLE_PATH)
#   plt.figure(0, figsize=(10, 5))
#   plt.imshow(raw_style)
#   plt.title("Original style image")
#   plt.show()
#
    style_image = raw_style.astype(np.float)
    style_shape = (1,) + style_image.shape # (h, w, nch) =>  (1, h, w, nch) 
    style_features = dict()
    with tf.Graph().as_default(), tf.Session() as sess:
        image = tf.placeholder('float', shape=style_shape)
        nets, _, _ = VGG19(VGG19_PATH, image)
        style_image_pre = np.array([preprocess(style_image, content_mean_pixel)])
        for idx, layer in enumerate(STYLE_LAYERS):
            curr_features = nets[layer].eval(feed_dict={image: style_image_pre})
            curr_features_vec = np.reshape(curr_features, (-1, curr_features.shape[3]))
            gram = np.matmul(curr_features_vec.T, curr_features_vec) / curr_features_vec.size
            style_features.update({layer: gram})
            # Plot 
#           plt.figure(idx, figsize=(10, 5))
#           plt.matshow(curr_features[0, :, :, 0], cmap=plt.cm.gray, fignum=idx)
#           plt.title("%d style feature" % (idx))
#           plt.show()



    content_weight = 5
    style_weight = 10
    tv_weight = 100
    learning_rate =5.
    iterations = 1000

    def _tensor_size(tensor):
        from operator import mul
        #return reduce(mul, (d.value for d in tensor.get_shape()), 1)
        result = 1
        for x in (d.value for d in tensor.get_shape()):
            result = mul(result, x)
        return result


    with tf.Graph().as_default(), tf.Session() as sess, tf.device('/gpu:0'):
        initial = tf.random_normal(content_shape) * 0.256
        image2opt = tf.Variable(initial)
        nets, mean_pixel, _ = VGG19(VGG19_PATH, image2opt)
        
        # content loss
        content_loss = content_weight*(2*tf.nn.l2_loss(nets[CONTENT_LAYER] - content_features) / content_features.size)
        
        # style loss
        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = nets[style_layer]
            _, height, width, number = layer.get_shape()
            size = height*width*number
            feats = tf.reshape(layer, (-1, number.value))
            gram = tf.matmul(tf.transpose(feats), feats) / size.value
            style_gram = style_features[style_layer]
            style_losses.append(2*tf.nn.l2_loss(gram - style_gram) / style_gram.size)
            
        style_loss = style_weight*tf.reduce_sum(style_losses)
        
        # Total variation denoising
        tv_y_size = _tensor_size(image2opt[:,1:,:,:])
        tv_x_size = _tensor_size(image2opt[:,:,1:,:])
        tv_loss = tv_weight*2*(
            (tf.nn.l2_loss(image2opt[:,1:,:,:] - image2opt[:,:content_shape[1]-1,:,:]) / tv_y_size) +
            (tf.nn.l2_loss(image2opt[:,:,1:,:] - image2opt[:,:,:content_shape[2]-1,:]) / tv_x_size)
        )
        
        # Overall loss
        loss = content_loss + style_loss + tv_loss
        
        # Optimizer : l-bfgs is more suitable if it is possible
        trainer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())
        for i in range(iterations):
            trainer.run()
            if i % 100 == 0 or i == iterations-1:
                #print('{}/{}'.format(i, iterations))
                cr += '{}/{}'.format(i, iterations) + delim
                out = image2opt.eval()
                stylized_img = out[0, :, :, :] + content_mean_pixel
                stylized_img = np.clip(stylized_img, 0, 255).astype('uint8')
#               plt.figure(0, figsize=(10,5))
#               plt.imshow(stylized_img)
#               plt.title('{:d}th stylized image'.format(i))
#               plt.show()
        out = image2opt.eval()


#   plt.figure(0, figsize=(10, 5))
#   plt.imshow(raw_content)
#   plt.title("Original content image")
#   plt.show()

#   plt.figure(0, figsize=(10, 5))
#   plt.imshow(raw_style)
#   plt.title("Original style image")
#   plt.show()
#
    stylized_img = out[0, :, :, :] + content_mean_pixel
    stylized_img = np.clip(stylized_img, 0, 255).astype('uint8')
#   plt.figure(1, figsize=(10, 5))
#   plt.imshow(stylized_img)
#   plt.title("Stylized image")
#   plt.show()

    fig = plt.figure(figsize=(10,15))
    #plt.axes('off')
    plt.subplot(311)
    plt.imshow(raw_content[::-1,:,:])
    plt.subplot(312)
    plt.imshow(raw_style[::-1,:,:])
    plt.subplot(313)
    plt.imshow(stylized_img[::-1,:,:])
    plt.imsave('neural_stylized_image_sample.jpg', stylized_img)

    spent = (time.time() - start) / 60.
    cr += 'TIME : {:.2f} min'.format(spent) + delim
    return [cr, mpld3.fig_to_html(fig)]


if __name__ == '__main__':
    img = styler_transfer()
