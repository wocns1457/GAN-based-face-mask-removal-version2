import os
import tensorflow as tf
from tqdm import tqdm
from datasets import Dataset
from utils import *

tf.random.set_seed(42)  

class Train_Mask:
    def __init__(self, model, alpha=100, lr=2e-4, checkpoint_dir=None):
        self.model = model
        self.model.build(input_shape=(None, 128, 128, 3))
        self.alpha = alpha
        self.lr = lr
        self.checkpoint_dir = checkpoint_dir
        self.bce_loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.5)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.optimizer,
                                        generator=self.model)
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        
    def generator_loss(self, gen_output, target):
        # Binary cross entropy
        bce_loss = self.bce_loss(target, gen_output)

        # l1_loss = self.alpha * tf.reduce_mean(tf.abs(target - gen_output))
        # total_gen_loss = gan_loss + (self.alpha * l1_loss)

        return bce_loss

    # @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape:
            gen_output = self.model(input_image, training=True)
            gen_loss = self.generator_loss(gen_output, target)
        generator_gradients = gen_tape.gradient(gen_loss,
                                                self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(generator_gradients,
                                            self.model.trainable_variables))
        
        return gen_loss.numpy()
    
    def save(self, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.checkpoint.save(file_prefix= self.checkpoint_prefix)

    def load(self, checkpoint_dir):
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        
    def fit(self, dataset, epochs):     
        for epoch in range(epochs):
            pbar = tqdm(enumerate(dataset), total=len(dataset), desc='epoch', ncols=80)   
            pbar.set_description(f'{epoch+1} epoch')
            for step, (_, mask_input ,binary_input) in pbar:
                binary_input = np.where(binary_input >= 0.5, 1, 0)
                binary_input = tf.convert_to_tensor(binary_input, dtype=tf.float32)
                loss = self.train_step(mask_input, binary_input[:,:,:, 0])
                # and pbar loss update
                if step % 20 == 0:
                    pbar.set_postfix(loss=loss)    
                # training_visualization once per 500 step
                if step % 1000 == 0:
                    training_visualization(self.model, mask_input, binary_input, epoch, step, task='mask')     
            # Save (checkpoint) the model once per 2 epcoh
            if (epoch + 1) % 2 == 0:
                self.save(self.checkpoint_dir)

class Train_Face:
    def __init__(self, mask_model, face_model, dis_whole_model, dis_region_model, vgg19, lr=2e-4, mask_checkpoint_dir=None, face_checkpoint_dir=None, dis_checkpoint_dir=None):
        self.mask_model = mask_model
        self.face_model = face_model
        self.dis_whole_model = dis_whole_model
        self.dis_region_model = dis_region_model
        self.vgg19 = vgg19.get_vgg19()

        self.mask_model.build(input_shape=(None, 128, 128, 3))
        self.face_model.build(input_shape=[(None, 128, 128, 3), (None, 128, 128, 1)])  
        self.dis_whole_model.build(input_shape=(None, 128, 128, 3))
        self.dis_region_model.build(input_shape=[(None, 128, 128, 3), (None, 128, 128, 1), (None, 128, 128, 3)])

        self.LAMBDA_whole = 0.3
        self.LAMBDA_mask = 0.7
        self.LAMBDA_rc = 100

        self.lr = lr
        self.gan_BCE_loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.mask_optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.5)
        self.face_optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.5)
        self.dis_whole_optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.5)
        self.dis_region_optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.5)

        self.mask_checkpoint_dir = mask_checkpoint_dir
        self.face_checkpoint_dir = face_checkpoint_dir
        self.dis_checkpoint_dir = dis_checkpoint_dir

        self.face_checkpoint_prefix = os.path.join(self.face_checkpoint_dir, "ckpt")
        self.dis_checkpoint_prefix = os.path.join(self.dis_checkpoint_dir, "ckpt")
        
        self.mask_checkpoint = tf.train.Checkpoint(generator_optimizer=self.mask_optimizer,
                                generator=self.mask_model)
                
        self.face_checkpoint = tf.train.Checkpoint(generator_optimizer=self.face_optimizer,
                                                  generator=self.face_model)
        
        self.dis_checkpoint = tf.train.Checkpoint(dis_whole_optimizer=self.dis_whole_optimizer,
                                          dis_region_optimizer=self.dis_region_optimizer, 
                                          dis_whole=self.dis_whole_model,
                                          dis_region=self.dis_region_model)

        self.mask_checkpoint.restore(tf.train.latest_checkpoint(self.mask_checkpoint_dir))
        self.face_checkpoint.restore(tf.train.latest_checkpoint(self.face_checkpoint_dir))
        self.dis_checkpoint.restore(tf.train.latest_checkpoint(self.dis_checkpoint_dir))

    # @tf.function
    def perceptual_loss(self, gen_image, gt_image):
        h1_list = self.vgg19(gen_image)
        h2_list = self.vgg19(gt_image)
        perc_loss = 0.0
        for h1, h2 in zip(h1_list, h2_list):
            h1 = tf.reshape(h1, [h1.shape[0], -1])
            h2 = tf.reshape(h2, [h2.shape[0], -1])
            perc_loss += tf.math.reduce_sum(tf.math.square((h1 - h2)), axis=-1)
        perc_loss = tf.reduce_mean(perc_loss)
        return perc_loss

    def gen_loss(self, disc_gen_output):
        adv_loss = self.gan_BCE_loss(tf.ones_like(disc_gen_output), disc_gen_output) # Adversarial loss
        return adv_loss
        
    def rec_loss(self, gen_output, target):
        # ssim loss
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(gen_output, target, max_val=1.0))
        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        rec_loss = ssim_loss + l1_loss
        return rec_loss

    def disc_loss(self, disc_real_output, disc_gen_output):
        real_loss = self.gan_BCE_loss(tf.ones_like(disc_real_output), disc_real_output) # Real samples
        fake_loss = self.gan_BCE_loss(tf.zeros_like(disc_gen_output), disc_gen_output) # Fake samples
        total_loss = real_loss + fake_loss
        return total_loss

    # @tf.function
    def train_step_first(self, input_image, input_map, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_image = self.face_model([input_image, input_map], training=True)

            real_output = self.dis_whole_model(input_image, training=True)
            fake_output = self.dis_whole_model(generated_image, training=True)
        
            generator_loss = self.gen_loss(fake_output)
            discriminator_loss = self.disc_loss(real_output, fake_output)
            rc_loss = self.rec_loss(generated_image, target)
            perc_loss = self.perceptual_loss(generated_image, target)
            gen_tot_loss = self.LAMBDA_rc*(rc_loss + perc_loss) + generator_loss
        
        gradients_generator = gen_tape.gradient(gen_tot_loss,
                                            self.face_model.trainable_variables)
        gradients_disc_whole  = disc_tape.gradient(discriminator_loss,
                                            self.dis_whole_model.trainable_variables)
            
        self.face_optimizer.apply_gradients(zip(gradients_generator,
                                              self.face_model.trainable_variables))
        self.dis_whole_optimizer.apply_gradients(zip(gradients_disc_whole,
                                              self.dis_whole_model.trainable_variables))
        
        
        return gen_tot_loss.numpy(), discriminator_loss.numpy()


    # @tf.function
    def train_step_second(self, input_image, input_map, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_whole_tape, tf.GradientTape() as disc_mask_tape:
            generated_image = self.face_model([input_image, input_map], training=True)

            real_output_whole = self.dis_whole_model(input_image, training=True)
            fake_output_whole = self.dis_whole_model(generated_image, training=True)
            real_output_mask = self.dis_region_model([target, input_map, input_image], training=True)
            fake_output_mask = self.dis_region_model([generated_image, input_map, input_image], training=True)

            gen_loss_whole = self.gen_loss(fake_output_whole)
            disc_loss_whole = self.LAMBDA_whole * self.disc_loss(real_output_whole, fake_output_whole)
            gen_loss_mask = self.gen_loss(fake_output_mask)
            disc_loss_mask = self.LAMBDA_mask * self.disc_loss(real_output_mask, fake_output_mask)
            rc_loss = self.rec_loss(generated_image, target)
            perc_loss = self.perceptual_loss(generated_image, target)
            gen_tot_loss = self.LAMBDA_rc*(rc_loss + perc_loss) + self.LAMBDA_whole*(gen_loss_whole) + self.LAMBDA_mask*(gen_loss_mask)
        
        gradients_generator = gen_tape.gradient(gen_tot_loss, self.face_model.trainable_variables)
        gradients_disc_whole = disc_whole_tape.gradient(disc_loss_whole, self.dis_whole_model.trainable_variables)
        gradients_disc_mask = disc_mask_tape.gradient(disc_loss_mask, self.dis_region_model.trainable_variables)
            
        self.face_optimizer.apply_gradients(zip(gradients_generator,
                                              self.face_model.trainable_variables))
        self.dis_whole_optimizer.apply_gradients(zip(gradients_disc_whole,
                                              self.dis_whole_model.trainable_variables))
        self.dis_whole_optimizer.apply_gradients(zip(gradients_disc_mask,
                                      self.dis_region_model.trainable_variables))
        
        return gen_tot_loss.numpy(), disc_loss_whole.numpy(), disc_loss_mask.numpy()

    def save(self, face_checkpoint_dir, dis_checkpoint_dir):
        if not os.path.exists(face_checkpoint_dir):
            os.mkdir(face_checkpoint_dir)
        if not os.path.exists(dis_checkpoint_dir):
            os.mkdir(dis_checkpoint_dir)
        self.face_checkpoint.save(file_prefix= self.face_checkpoint_prefix)
        self.dis_checkpoint.save(file_prefix= self.dis_checkpoint_prefix)

    def load(self, face_checkpoint_dir, dis_checkpoint_dir):
        self.face_checkpoint.restore(tf.train.latest_checkpoint(face_checkpoint_dir))
        self.dis_checkpoint.restore(tf.train.latest_checkpoint(dis_checkpoint_dir))

    def fit(self, dataset, epochs):     
        ratio = int(round(epochs * 0.4))
        for epoch in range(epochs):
            pbar = tqdm(enumerate(dataset), total=len(dataset), desc='epoch', ncols=150)   
            pbar.set_description(f'{epoch+1} epoch')
            disc_region_loss = 0
            for step, (real_input, mask_input, binary_input) in pbar:
                if epoch < ratio:
                    gen_output = self.mask_model(mask_input, training=False)
                    binary_input = noise_processing(gen_output)
                    gen_tot_loss, disc_whole_loss = self.train_step_first(mask_input, binary_input, real_input)
                else:
                    gen_output = self.mask_model(mask_input, training=False)
                    binary_input = noise_processing(gen_output)
                    gen_tot_loss, disc_whole_loss, disc_region_loss = self.train_step_second(mask_input, binary_input, real_input)
                
                # # and pbar loss update
                if step % 20 == 0:
                    pbar.set_postfix(gen_loss=gen_tot_loss, whole_loss=disc_whole_loss, region_loss=disc_region_loss)    
                # training_visualization once per 500 step
                if step % 500 == 0:
                    face_training_visualization(self.face_model, mask_input, binary_input, real_input, epoch, step)     
            # Save (checkpoint) the model once per 2 epcoh
            if (epoch + 1) % 1 == 0:
                self.save(self.face_checkpoint_dir, self.dis_checkpoint_dir)
              


##################################################################
# config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.95)
# #device_count = {'GPU': 1}
# )
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)
###################################################################
# gpu = tf.config.experimental.list_physical_devices('GPU') # 내 컴에 장착된 GPU를 list로 반환
# try:
#     tf.config.experimental.set_memory_growth(gpu[0], True) # GPU Memory Growth를 Enable
# except RuntimeError as e:
#     print(e) # Error 발생하면 Error 내용 출력
###################################################################
# PATH = os.getcwd()
# train_path = PATH+'/train'                
# mask_checkpoint_dir = PATH+'/mask32_checkpoints'
# face_checkpoint_dir = PATH+'/face32_checkpoints'

# BATCH_SIZE = 4
# trainset = Dataset(file_path=train_path, batch_size=BATCH_SIZE, task='mask')
# trainset = trainset.make_train()
# with tf.device('/gpu:0'):
                ######  mask train ############
    # mask_G = Mask_G(filters=32)
    # mask_G_train = Train_Mask(mask_G, checkpoint_dir=mask_checkpoint_dir)
    # # mask_G_train.load(checkpoint_dir=mask_checkpoint_dir)
    # mask_G_train.fit(trainset, epochs=4)


                #########  Face train ############
    # mask_G, face_G, face_D = Mask_G(filters=32), Face_G(filters=32), Face_D(filters=32)
    # face_train = Train_Face(mask_G, face_G, face_D, 
    #                 mask_checkpoint_dir=mask_checkpoint_dir, face_checkpoint_dir=face_checkpoint_dir)
    # face_train.fit(trainset, epochs=4)
    # face_train.load(face_checkpoint_dir)
    # print(face_train.face_model.get_weights(),'\n')


