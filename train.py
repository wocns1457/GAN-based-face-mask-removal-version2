import os
from tqdm import tqdm
import tensorflow as tf
from datasets import Dataset
from utils import *

tf.random.set_seed(42)  

class Train_Model():
    def __init__(self, mask_model, face_model, dis_whole_model, dis_region_model, vgg19, lr=2e-4, mask_checkpoint_dir=None, face_checkpoint_dir=None, dis_checkpoint_dir=None):
        self.mask_model = mask_model
        self.face_model = face_model
        self.dis_whole_model = dis_whole_model
        self.dis_region_model = dis_region_model
        self.vgg19 = vgg19.get_vgg19()

        self.LAMBDA_whole = 0.3
        self.LAMBDA_mask = 0.7
        self.LAMBDA_rc = 100

        self.BCE_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.mask_optimizer = tf.keras.optimizers.legacy.Adam(lr, beta_1=0.5)
        self.face_G_optimizer = tf.keras.optimizers.legacy.Adam(lr, beta_1=0.5)
        self.face_D_optimizer = tf.keras.optimizers.legacy.Adam(lr, beta_1=0.5)

        self.mask_checkpoint_dir = mask_checkpoint_dir
        self.face_checkpoint_dir = face_checkpoint_dir
        self.dis_checkpoint_dir = dis_checkpoint_dir

        self.mask_checkpoint_prefix = os.path.join(self.mask_checkpoint_dir, "ckpt")
        self.face_checkpoint_prefix = os.path.join(self.face_checkpoint_dir, "ckpt")
        self.dis_checkpoint_prefix = os.path.join(self.dis_checkpoint_dir, "ckpt")

        self.mask_checkpoint = tf.train.Checkpoint(generator_optimizer=self.mask_optimizer,
                                                    generator=self.mask_model)

        self.face_checkpoint = tf.train.Checkpoint(generator_optimizer=self.face_G_optimizer,
                                                    generator=self.face_model)

        self.dis_checkpoint = tf.train.Checkpoint(generator_optimizer=self.face_D_optimizer,
                                                    dis_whole=self.dis_whole_model,
                                                    dis_region=self.dis_region_model)

        self.mask_checkpoint.restore(tf.train.latest_checkpoint(self.mask_checkpoint_dir))
        self.face_checkpoint.restore(tf.train.latest_checkpoint(self.face_checkpoint_dir))
        self.dis_checkpoint.restore(tf.train.latest_checkpoint(self.dis_checkpoint_dir))

    def mask_loss(self, gen_output, target):
        gen_output = (gen_output + 1) / 2
        target = (target + 1) / 2
        return self.BCE_loss(target, gen_output)

    # @tf.function
    def perceptual_loss(self, gen_image, gt_image):
        h1_list = self.vgg19(gt_image)
        h2_list = self.vgg19(gen_image)
        perc_loss = 0.0
        for h1, h2 in zip(h1_list, h2_list):
            h1 = tf.reshape(h1, [h1.shape[0], -1])
            h2 = tf.reshape(h2, [h2.shape[0], -1])
            perc_loss += tf.math.reduce_mean(tf.math.square((h1 - h2)), axis=-1)
        perc_loss = tf.reduce_mean(perc_loss)
        return perc_loss

    def rc_loss(self, gen_output, target):
        # ssim loss
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(gen_output, target, max_val=1.0))
        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        return ssim_loss + l1_loss

    def disc_loss(self, disc_real_output, disc_gen_output):
        real_loss = self.BCE_loss(tf.ones_like(disc_real_output), disc_real_output) # Real samples
        fake_loss = self.BCE_loss(tf.zeros_like(disc_gen_output), disc_gen_output) # Fake samples
        return real_loss + fake_loss

    def adv_loss(self, disc_gen_output):
        return self.BCE_loss(tf.ones_like(disc_gen_output), disc_gen_output)

    def mask_train_step(self, input_image, target):
        with tf.GradientTape() as mask_tape:
            mask_map = self.mask_model(input_image, training=True)

            mask_loss = self.mask_loss(mask_map, target)

        gradients_mask = mask_tape.gradient(mask_loss,
                                            self.mask_model.trainable_variables)
        self.mask_optimizer.apply_gradients(zip(gradients_mask,
                                              self.mask_model.trainable_variables))
        return mask_map, mask_loss.numpy()

    def face_train_first_step(self, input_image, mask_map, target):
        with tf.GradientTape() as face_G_tape:
            generated_image = self.face_model([input_image, mask_map], training=True)

            rc_loss = self.rc_loss(generated_image, target)

            face_G_total_loss = rc_loss

        gradients_face_G  = face_G_tape.gradient(face_G_total_loss,
                                            self.face_model.trainable_variables)
        self.face_G_optimizer.apply_gradients(zip(gradients_face_G,
                                              self.face_model.trainable_variables))
        return face_G_total_loss.numpy()

    def face_train_second_step(self, input_image, mask_map, target):
        with tf.GradientTape() as face_G_tape, tf.GradientTape() as face_D_tape:
            generated_image = self.face_model([input_image, mask_map], training=True)

            real_output = self.dis_whole_model([generated_image, target], training=True)
            fake_output = self.dis_whole_model(generated_image, training=True)

            rc_loss = self.rc_loss(generated_image, target)
            perc_loss = self.perceptual_loss(generated_image, target)
            adv_loss = self.adv_loss(fake_output)
            disc_loss = self.disc_loss(real_output, fake_output)

            face_G_total_loss = self.LAMBDA_rc*(rc_loss + perc_loss) + adv_loss

            face_D_total_loss = disc_loss

        gradients_face_G = face_G_tape.gradient(face_G_total_loss,
                                                self.face_model.trainable_variables)
        self.face_G_optimizer.apply_gradients(zip(gradients_face_G,
                                                  self.face_model.trainable_variables))

        gradients_face_D = face_D_tape.gradient(face_D_total_loss,
                                                    self.dis_whole_model.trainable_variables)
        self.face_D_optimizer.apply_gradients(zip(gradients_face_D,
                                                  self.dis_whole_model.trainable_variables))

        return face_G_total_loss.numpy(), face_D_total_loss.numpy()

    def face_train_third_step(self, input_image, mask_map, target):
        with tf.GradientTape() as face_G_tape, tf.GradientTape() as face_D_tape:
            generated_image = self.face_model([input_image, mask_map], training=True)

            whole_real_output = self.dis_whole_model([generated_image, target], training=True)
            whole_fake_output = self.dis_whole_model(generated_image, training=True)
            mask_region_real_output_mask = self.dis_region_model([input_image, mask_map, generated_image, target], training=True)
            mask_region_fake_output_mask = self.dis_region_model(generated_image, training=True)

            rc_loss = self.rc_loss(generated_image, target)
            perc_loss = self.perceptual_loss(generated_image, target)

            whole_disc_loss = self.disc_loss(whole_real_output, whole_fake_output)
            mask_region_disc_loss = self.disc_loss(mask_region_real_output_mask, mask_region_fake_output_mask)
            whole_adv_loss = self.adv_loss(whole_fake_output)
            mask_region_adv_loss = self.adv_loss(mask_region_fake_output_mask)

            face_G_total_loss = self.LAMBDA_rc*(rc_loss + perc_loss) + self.LAMBDA_whole*whole_adv_loss + self.LAMBDA_mask*mask_region_adv_loss

            face_D_total_loss = self.LAMBDA_whole*whole_disc_loss + self.LAMBDA_mask*mask_region_disc_loss

        gradients_face_G = face_G_tape.gradient(face_G_total_loss,
                                                 self.face_model.trainable_variables)
        self.face_G_optimizer.apply_gradients(zip(gradients_face_G,
                                                  self.face_model.trainable_variables))

        gradients_face_D = face_D_tape.gradient(face_D_total_loss,
                                                  self.dis_whole_model.trainable_variables +
                                                  self.dis_region_model.trainable_variables)
        self.face_D_optimizer.apply_gradients(zip(gradients_face_D,
                                                  self.dis_whole_model.trainable_variables +
                                                  self.dis_region_model.trainable_variables))

        return face_G_total_loss.numpy(), face_D_total_loss.numpy()

    def save(self, mask_checkpoint_dir, face_checkpoint_dir, dis_checkpoint_dir):
        if not os.path.exists(mask_checkpoint_dir):
            os.mkdir(mask_checkpoint_dir)
        if not os.path.exists(face_checkpoint_dir):
            os.mkdir(face_checkpoint_dir)
        if not os.path.exists(dis_checkpoint_dir):
            os.mkdir(dis_checkpoint_dir)
        self.mask_checkpoint.save(file_prefix=self.mask_checkpoint_prefix)
        self.face_checkpoint.save(file_prefix=self.face_checkpoint_prefix)
        self.dis_checkpoint.save(file_prefix=self.dis_checkpoint_prefix)

    def load(self, m_ckpt_num=None, f_ckpt_num=None, d_ckpt_num=None):
        if m_ckpt_num is None:
            self.mask_checkpoint.restore(tf.train.latest_checkpoint(self.mask_checkpoint_dir))
        else:
            self.mask_checkpoint.restore(self.mask_checkpoint_dir+'/ckpt-{m_ckpt_num}'.format(m_ckpt_num=m_ckpt_num))
        if f_ckpt_num is None:
            self.face_checkpoint.restore(tf.train.latest_checkpoint(self.face_checkpoint_dir))
        else:
            self.face_checkpoint.restore(self.face_checkpoint_dir+'/ckpt-{f_ckpt_num}'.format(f_ckpt_num=f_ckpt_num))
        if d_ckpt_num is None:
            self.dis_checkpoint.restore(tf.train.latest_checkpoint(self.dis_checkpoint_dir))
        else:
            self.dis_checkpoint.restore(self.dis_checkpoint_dir+'/ckpt-{d_ckpt_num}'.format(d_ckpt_num=d_ckpt_num))

    def fit(self, dataset, epochs):
        first_step_ratio, second_step_ratio = int(round(epochs * 0.1)), int(round(epochs * 0.4))
        for epoch in range(1, epochs+1):
            face_D_loss = 0.0
            pbar = tqdm(enumerate(dataset), total=len(dataset), desc='epoch', ncols=150)
            pbar.set_description(f'{epoch} epoch')
            for step, (real_input, mask_input, binary_input) in pbar:
                binary_input = binary_input[:, :, :, 0]
                mask_map, mask_loss = self.mask_train_step(mask_input, binary_input)
                mask_map_noise_processing = noise_processing(mask_map)

                if epoch < first_step_ratio:
                    face_G_loss = self.face_train_first_step(mask_input, mask_map_noise_processing, real_input)
                elif epoch < second_step_ratio:
                    face_G_loss, face_D_loss = self.face_train_second_step(mask_input, mask_map_noise_processing, real_input)
                else:
                    face_G_loss, face_D_loss = self.face_train_third_step(mask_input, mask_map_noise_processing, real_input)

                if step % 2 == 0:
                    pbar.set_postfix(mask_loss=mask_loss, face_G_loss=face_G_loss, face_D_loss=face_D_loss)
                # training_visualization once per 500 step
                if step % 500 == 0:
                    face_training_visualization(self.face_model, mask_input, mask_map_noise_processing, real_input, epoch, step+1)
            # Save (checkpoint) the model once per 10 epoch
            if epoch % 10 == 0:
                self.save(self.mask_checkpoint_dir, self.face_checkpoint_dir, self.dis_checkpoint_dir)
