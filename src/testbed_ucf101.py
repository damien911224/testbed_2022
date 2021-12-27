import tensorflow as tf
import os
import sys
import numpy as np
import glob
import json
import cv2
import random
import math
import time
import I3D
import gc
import matplotlib
import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from tensorflow.python.ops import math_ops
from shutil import rmtree, copyfile
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.manifold import TSNE
import itertools


class Networks:

    def __init__(self):
        self.input_size = (224, 224, 3)

    def pretrain(self):
        print("=" * 90)
        print("Networks Training")
        print("=" * 90)

        self.is_server = True
        self.batch_size = 4 if self.is_server else 2
        self.num_gpus = 4 if self.is_server else 1
        self.num_workers = self.num_gpus * 24
        self.data_type = "images"
        self.dataset_name = "ucf101"
        self.dataset_split = "split01"
        self.flow_type = "tvl1"
        self.optimizer_type = "SGD"
        if self.dataset_name == "ucf101":
            self.epochs = 25
        else:
            self.epochs = 25
        self.temporal_width = 64
        self.display_term = 1
        self.dtype = tf.float32
        self.dformat = "NDHWC"

        self.model_name = "I3D"
        now = time.localtime()
        self.train_date = "{:02d}{:02d}".format(now.tm_mon, now.tm_mday)

        self.validation_batch_size = self.batch_size
        self.validation_term = 1
        self.validation_temporal_width = self.temporal_width
        self.validation_display_term = self.display_term
        self.ckpt_save_term = 5

        self.dataset = self.Dataset(self)

        self.train_data, self.validation_data = self.dataset.getDataset("train")
        self.train_iterator = self.train_data.tf_dataset.make_initializable_iterator()
        self.train_next_element = self.train_iterator.get_next()

        self.validation_iterator = self.validation_data.tf_dataset.make_one_shot_iterator()
        self.validation_next_element = self.validation_iterator.get_next()
        self.validation_size = self.validation_data.data_count // 1

        self.save_ckpt_file_folder = \
            os.path.join(self.dataset.root_path,
                         "networks", "weights",
                         "save", "{}_{}_{}_{}_{}".format(self.model_name,
                                                         self.dataset_name.upper(),
                                                         "RGB" if self.data_type == "images" else "Flow",
                                                         "Pretraining",
                                                         self.train_date))

        self.summary_folder = os.path.join(self.dataset.root_path,
                                           "networks", "summaries",
                                           "{}_{}_{}_{}_{}".format(self.model_name,
                                                                   self.dataset_name.upper(),
                                                                   "RGB" if self.data_type == "images" else "Flow",
                                                                   "Pretraining",
                                                                   self.train_date))
        self.train_summary_file_path = os.path.join(self.summary_folder, "train_summary")
        self.validation_summary_file_path = os.path.join(self.summary_folder, "validation_summary")

        self.global_step = tf.Variable(0, trainable=False)
        self.global_epochs = tf.Variable(1, trainable=False)
        if self.optimizer_type == "Adam":
            self.starter_learning_rate = 2.0e-4
        else:
            self.starter_learning_rate = 1.0e-2

        if self.dataset_name == "ucf101":
            boundaries = [20, 23]
        else:
            boundaries = [20, 23]
        values = [self.starter_learning_rate,
                  self.starter_learning_rate * 1.0e-1,
                  self.starter_learning_rate * 1.0e-2]
        self.learning_rate = tf.train.piecewise_constant(self.global_epochs, boundaries, values)

        global current_learning_rate
        current_learning_rate = list()

        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                    momentum=0.9)

        self.model = self.Model(self, is_training=True, phase="pretraining", data_type=self.data_type)
        self.model_validation = self.Model(self, is_training=False, phase="pretraining", data_type=self.data_type)
        self.model.build_model()
        self.model_validation.build_model()

        self.parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        self.parameter_dict = dict()
        for parameter in self.parameters:
            self.parameter_dict[parameter.name] = parameter

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            with tf.device("/cpu:0"):
                self.train_step = self.optimizer.apply_gradients(self.model.average_grads,
                                                                 global_step=self.global_step)

        with tf.device("/cpu:0"):
            for parameter in self.parameters:
                if "BatchNorm" not in parameter.name:
                    tf.summary.histogram(parameter.name, parameter)

            self.variable_summary = tf.summary.merge_all()

            if self.optimizer_type == "Adam":
                current_learning_rate = tf.reduce_mean(tf.stack(current_learning_rate))
            else:
                current_learning_rate = self.learning_rate

            self.loss_summary_ph = tf.placeholder(dtype=tf.float32)
            self.loss_summary = tf.summary.scalar("loss", self.loss_summary_ph)
            self.solver_loss_summary_ph = tf.placeholder(dtype=tf.float32)
            self.solver_loss_summary = tf.summary.scalar("solver_loss", self.solver_loss_summary_ph)
            self.reconstruction_loss_summary_ph = tf.placeholder(dtype=tf.float32)
            self.reconstruction_loss_summary = tf.summary.scalar("reconstruction", self.reconstruction_loss_summary_ph)
            self.current_learning_rate_ph = tf.placeholder(dtype=tf.float32)
            self.current_learning_rate_summary = tf.summary.scalar("current_learning_rate",
                                                                   self.current_learning_rate_ph)

            image_summary_size = 10 * 3
            self.image_summary_ph = \
                tf.placeholder(dtype=tf.uint8,
                               shape=(image_summary_size, 224, 224 * 2 + 10, 3))
            self.image_summary = \
                tf.summary.image("reconstruction_images",
                                 self.image_summary_ph,
                                 max_outputs=image_summary_size)

            self.train_summaries = tf.summary.merge([self.variable_summary,
                                                     self.loss_summary,
                                                     self.solver_loss_summary,
                                                     self.reconstruction_loss_summary,
                                                     self.current_learning_rate_summary])

            self.validation_summaries = tf.summary.merge([self.loss_summary,
                                                          self.solver_loss_summary,
                                                          self.reconstruction_loss_summary,
                                                          self.image_summary])

        os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(device_id) for device_id in range(self.num_gpus)])
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"

        self.best_validation = float("-inf")
        self.previous_best_epoch = None

        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=self.epochs)

        with tf.Session() as session:
            session.run(self.train_iterator.initializer)

            rmtree(self.summary_folder, ignore_errors=True)
            rmtree(self.save_ckpt_file_folder, ignore_errors=True)
            try:
                os.mkdir(self.save_ckpt_file_folder)
            except OSError:
                pass
            self.train_summary_writer = tf.summary.FileWriter(self.train_summary_file_path, session.graph)
            self.validation_summary_writer = tf.summary.FileWriter(self.validation_summary_file_path)

            # Initialize all the variables
            init_variables = tf.global_variables_initializer()
            session.run(init_variables)

            batch_iteration = 1

            summary = session.run(self.variable_summary)
            self.train_summary_writer.add_summary(summary, 0)

            for epoch in range(1, self.epochs + 1, 1):
                session.run([self.train_iterator.initializer, self.global_epochs.assign(epoch)])
                epoch_loss = 0.0
                epoch_solver_loss = 0.0
                epoch_reconstruction_loss = 0.0
                epoch_learning_rate = 0.0
                epoch_batch_iteration = 0
                epoch_time = 0.0
                epoch_preprocessing_time = 0.0
                epoch_training_time = 0.0

                batch_length = int(
                    math.floor(float(self.train_data.data_count) / float(self.batch_size * self.num_gpus)))

                while True:
                    iteration_start_time = time.time()
                    preprocessing_start_time = time.time()
                    try:
                        frame_vectors, target_vectors, masks = session.run(self.train_next_element)
                    except tf.errors.OutOfRangeError:
                        break

                    epoch_preprocessing_time += time.time() - preprocessing_start_time

                    train_step_start_time = time.time()
                    _, loss, \
                    solver_loss, \
                    reconstruction_loss, \
                    current_lr = \
                        session.run(
                            [self.train_step,
                             self.model.loss,
                             self.model.solver_loss,
                             self.model.reconstruction_loss,
                             current_learning_rate],
                            feed_dict={self.model.frames: frame_vectors,
                                       self.model.masks: masks})

                    epoch_training_time += time.time() - train_step_start_time
                    epoch_loss += loss
                    epoch_solver_loss += solver_loss
                    epoch_reconstruction_loss += reconstruction_loss
                    epoch_learning_rate += current_lr

                    print_string = \
                        "|{:10s}|Epoch {:3d}/{:3d}|Batch {:3d}/{:3d}|Loss: {:.2f}".format(
                            "Training",
                            epoch,
                            self.epochs,
                            epoch_batch_iteration + 1,
                            batch_length,
                            loss)
                    progress_step = epoch_batch_iteration + 1
                    progress_length = batch_length
                    print_string += \
                        " |{}{}|".format(
                            "=" * int(round(37.0 * float(progress_step) / float(progress_length))),
                            " " * (37 - int(round(37.0 * float(progress_step) / float(progress_length)))))
                    sys.stdout.write("\r" + print_string)
                    sys.stdout.flush()

                    epoch_batch_iteration += 1
                    batch_iteration += 1

                    epoch_time += time.time() - iteration_start_time

                print()
                epoch_loss /= float(epoch_batch_iteration)
                epoch_solver_loss /= float(epoch_batch_iteration)
                epoch_reconstruction_loss /= float(epoch_batch_iteration)
                epoch_learning_rate /= float(epoch_batch_iteration)
                epoch_training_time /= float(epoch_batch_iteration)
                epoch_preprocessing_time /= float(epoch_batch_iteration)

                train_summary = session.run(self.train_summaries,
                                            feed_dict={self.loss_summary_ph: epoch_loss,
                                                       self.solver_loss_summary_ph: epoch_solver_loss,
                                                       self.reconstruction_loss_summary_ph: epoch_reconstruction_loss,
                                                       self.current_learning_rate_ph: epoch_learning_rate
                                                       })
                self.train_summary_writer.add_summary(train_summary, epoch)

                print("=" * 90)
                print("Epoch {:05d} Done ... Current Batch Iterations {:07d}".format(epoch, batch_iteration))
                print("Epoch {:05d} Loss {:.8f}".format(epoch, epoch_loss))
                print("Epoch {:05d} Takes {:03d} Batch Iterations".format(epoch, epoch_batch_iteration))
                print("Epoch {:05d} Takes {:.2f} Hours".format(epoch, epoch_time / 3600.0))
                print("Epoch {:05d} Average One Loop Time {:.2f} Seconds".format(epoch,
                                                                                 epoch_time / float(
                                                                                     epoch_batch_iteration)))
                print("Epoch {:05d} Average One Preprocessing Time {:.2f} Seconds".format(epoch,
                                                                                          epoch_preprocessing_time))
                print("Epoch {:05d} Average One Train Step Time {:.2f} Seconds".format(epoch,
                                                                                       epoch_training_time))
                print("Epoch {:05d} Current Learning Rate {:f}".format(epoch, epoch_learning_rate))
                print("=" * 90)

                if (epoch) % self.validation_term == 0 or epoch == 1:
                    print("Validation on Epochs {:05d}".format(epoch))

                    validation_loss = 0.0
                    validation_solver_loss = 0.0
                    validation_reconstruction_loss = 0.0
                    reconstruction_images = list()

                    loop_rounds = max(int(math.ceil(float(self.validation_size) /
                                                    float(self.validation_batch_size * self.num_gpus))), 1)

                    for validation_batch_index in range(loop_rounds):
                        try:
                            frame_vectors, target_vectors, masks, identities = \
                                session.run(self.validation_next_element)
                        except tf.errors.OutOfRangeError:
                            break

                        loss, solver_loss, reconstruction_loss, \
                        reconstruction_predictions = \
                            session.run(
                                [self.model_validation.loss,
                                 self.model_validation.solver_loss,
                                 self.model_validation.reconstruction_loss,
                                 self.model_validation.reconstruction_predictions],
                                feed_dict={self.model_validation.frames: frame_vectors,
                                           self.model_validation.masks: masks})

                        validation_loss += loss
                        validation_solver_loss += solver_loss
                        validation_reconstruction_loss += reconstruction_loss

                        if validation_batch_index < 10:
                            sampled_indices = random.sample(range(len(frame_vectors)), 3)
                            for n_i in sampled_indices:
                                sampled_t = random.choice(range(self.temporal_width))
                                t_image = np.array(((frame_vectors[n_i, sampled_t] + 1.0) / 2.0) * 255.0,
                                                   dtype=np.uint8)
                                p_image = np.array(
                                    np.clip(((reconstruction_predictions[n_i, sampled_t] + 1.0) / 2.0) * 255.0,
                                            0.0, 255.0), dtype=np.uint8)
                                buffer = np.zeros(dtype=np.uint8, shape=(224, 10, 3))
                                image = np.concatenate([t_image, buffer, p_image], axis=1)
                                reconstruction_images.append(image)

                        print_string = \
                            "|{:10s}|Epoch {:3d}/{:3d}|Batch {:3d}/{:3d}|Loss: {:.2f}".format(
                                "Validation",
                                epoch,
                                self.epochs,
                                validation_batch_index + 1,
                                loop_rounds,
                                loss)
                        progress_step = validation_batch_index + 1
                        progress_length = loop_rounds
                        print_string += \
                            " |{}{}|".format(
                                "=" * int(round(37.0 * float(progress_step) / float(progress_length))),
                                " " * (37 - int(round(37.0 * float(progress_step) / float(progress_length)))))
                        sys.stdout.write("\r" + print_string)
                        sys.stdout.flush()

                    print()
                    validation_loss /= float(loop_rounds)
                    validation_solver_loss /= float(loop_rounds)
                    validation_reconstruction_loss /= float(loop_rounds)

                    validation_summary = \
                        session.run(self.validation_summaries,
                                    feed_dict={self.loss_summary_ph: validation_loss,
                                               self.solver_loss_summary_ph: validation_solver_loss,
                                               self.reconstruction_loss_summary_ph: validation_reconstruction_loss,
                                               self.image_summary_ph: reconstruction_images})
                    self.validation_summary_writer.add_summary(validation_summary, epoch)

                    validation_quality = -validation_loss

                    if epoch % self.ckpt_save_term == 0:
                        # if self.previous_best_epoch and self.previous_best_epoch != epoch - self.ckpt_save_term:
                        #     weight_files = glob.glob(os.path.join(self.save_ckpt_file_folder,
                        #                                           "weights.ckpt-{}.*".format(
                        #                                               epoch - self.ckpt_save_term)))
                        #     for file in weight_files:
                        #         try:
                        #             os.remove(file)
                        #         except OSError:
                        #             pass

                        saver.save(session, os.path.join(self.save_ckpt_file_folder, "weights.ckpt"),
                                   global_step=epoch)

                    if validation_quality >= self.best_validation:
                        self.best_validation = validation_quality
                        if self.previous_best_epoch and self.previous_best_epoch % self.ckpt_save_term != 0:
                            weight_files = glob.glob(os.path.join(self.save_ckpt_file_folder,
                                                                  "weights.ckpt-{}.*".format(self.previous_best_epoch)))
                            for file in weight_files:
                                try:
                                    os.remove(file)
                                except OSError:
                                    pass

                        if epoch % self.ckpt_save_term != 0:
                            saver.save(session, os.path.join(self.save_ckpt_file_folder, "weights.ckpt"),
                                       global_step=epoch)
                        self.previous_best_epoch = epoch

                    print("Validation Results ...")
                    print("Validation Loss {:.5f}".format(validation_loss))
                    print("=" * 90)

    def finetune(self):
        print("=" * 90)
        print("Networks Training")
        print("=" * 90)

        self.is_server = True
        self.batch_size = 16 if self.is_server else 2
        self.num_gpus = 4 if self.is_server else 1
        self.num_workers = self.num_gpus * 24
        self.data_type = "images"
        self.dataset_name = "ucf101"
        self.dataset_split = "split01"
        self.flow_type = "tvl1"
        self.optimizer_type = "SGD"
        if self.dataset_name == "ucf101":
            self.epochs = 25
        else:
            self.epochs = 25
        self.temporal_width = 16
        self.display_term = 1
        self.dtype = tf.float32
        self.dformat = "NCDHW"

        self.model_name = "I3D"
        now = time.localtime()
        self.train_date = "{:02d}{:02d}".format(now.tm_mon, now.tm_mday)

        self.validation_batch_size = self.batch_size
        self.validation_term = 1
        self.validation_temporal_width = self.temporal_width
        self.validation_display_term = self.display_term
        self.ckpt_save_term = 5

        self.dataset = self.Dataset(self)

        self.train_data, self.validation_data = self.dataset.getDataset("train")
        self.train_iterator = self.train_data.tf_dataset.make_initializable_iterator()
        self.train_next_element = self.train_iterator.get_next()

        self.validation_iterator = self.validation_data.tf_dataset.make_one_shot_iterator()
        self.validation_next_element = self.validation_iterator.get_next()
        self.validation_size = self.validation_data.data_count // 1

        self.save_ckpt_file_folder = \
            os.path.join(self.dataset.root_path,
                         "networks", "weights",
                         "save", "{}_{}_{}_{}".format(self.model_name,
                                                      self.dataset_name,
                                                      self.dataset_split,
                                                      self.train_date))
        self.load_ckpt_file_path = \
            os.path.join(self.dataset.root_path,
                         "networks", "weights",
                         "save", "{}_{}_{}_{}_{}".format(self.model_name,
                                                         self.dataset_name.upper(),
                                                         "RGB" if self.data_type == "images" else "Flow",
                                                         "Pretraining",
                                                         "1225"),
                         "weights.ckpt-{}".format(25))

        self.save_ckpt_file_folder = \
            os.path.join(self.dataset.root_path,
                         "networks", "weights",
                         "save", "{}_{}_{}_{}_{}".format(self.model_name,
                                                         self.dataset_name.upper(),
                                                         "RGB" if self.data_type == "images" else "Flow",
                                                         "Finetuing",
                                                         self.train_date))

        self.summary_folder = os.path.join(self.dataset.root_path,
                                           "networks", "summaries",
                                           "{}_{}_{}_{}_{}".format(self.model_name,
                                                                   self.dataset_name.upper(),
                                                                   "RGB" if self.data_type == "images" else "Flow",
                                                                   "Finetuing",
                                                                   self.train_date))
        self.train_summary_file_path = os.path.join(self.summary_folder, "train_summary")
        self.validation_summary_file_path = os.path.join(self.summary_folder, "validation_summary")

        self.global_step = tf.Variable(0, trainable=False)
        self.global_epochs = tf.Variable(1, trainable=False)
        if self.optimizer_type == "Adam":
            self.starter_learning_rate = 2.0e-4
        else:
            self.starter_learning_rate = 1.0e-2

        if self.dataset_name == "ucf101":
            boundaries = [20, 23]
        else:
            boundaries = [20, 23]
        values = [self.starter_learning_rate,
                  self.starter_learning_rate * 1.0e-1,
                  self.starter_learning_rate * 1.0e-2]
        self.learning_rate = tf.train.piecewise_constant(self.global_epochs, boundaries, values)

        global current_learning_rate
        current_learning_rate = list()

        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                    momentum=0.9)

        self.model = self.Model(self, is_training=True, phase="finetuning", data_type=self.data_type)
        self.model_validation = self.Model(self, is_training=False, phase="finetuning", data_type=self.data_type)
        self.model.build_model()
        self.model_validation.build_model()

        self.parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        load_parameters = dict()
        for param in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_name):
            if "Logits" not in param.name:
                key_name = param.name[:-2]
                load_parameters[key_name] = param

        self.parameter_dict = dict()
        for parameter in self.parameters:
            self.parameter_dict[parameter.name] = parameter

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            with tf.device("/cpu:0"):
                self.train_step = self.optimizer.apply_gradients(self.model.average_grads,
                                                                 global_step=self.global_step)

        with tf.device("/cpu:0"):
            for parameter in self.parameters:
                if "BatchNorm" not in parameter.name:
                    tf.summary.histogram(parameter.name, parameter)

            self.variable_summary = tf.summary.merge_all()

            if self.optimizer_type == "Adam":
                current_learning_rate = tf.reduce_mean(tf.stack(current_learning_rate))
            else:
                current_learning_rate = self.learning_rate

            self.loss_summary_ph = tf.placeholder(dtype=tf.float32)
            self.loss_summary = tf.summary.scalar("loss", self.loss_summary_ph)
            self.accuracy_summary_ph = tf.placeholder(dtype=tf.float32)
            self.accuracy_summary = tf.summary.scalar("accuracy", self.accuracy_summary_ph)
            self.current_learning_rate_ph = tf.placeholder(dtype=tf.float32)
            self.current_learning_rate_summary = tf.summary.scalar("current_learning_rate",
                                                                   self.current_learning_rate_ph)

            self.train_summaries = tf.summary.merge([self.variable_summary, self.loss_summary,
                                                     self.accuracy_summary,
                                                     self.current_learning_rate_summary])

            self.validation_summaries = tf.summary.merge([self.loss_summary,
                                                          self.accuracy_summary])

        os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(device_id) for device_id in range(self.num_gpus)])
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"

        self.best_validation = float("-inf")
        self.previous_best_epoch = None

        loader = tf.train.Saver(var_list=load_parameters)
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=self.epochs)

        with tf.Session() as session:
            session.run(self.train_iterator.initializer)

            rmtree(self.summary_folder, ignore_errors=True)
            rmtree(self.save_ckpt_file_folder, ignore_errors=True)
            try:
                os.mkdir(self.save_ckpt_file_folder)
            except OSError:
                pass
            self.train_summary_writer = tf.summary.FileWriter(self.train_summary_file_path, session.graph)
            self.validation_summary_writer = tf.summary.FileWriter(self.validation_summary_file_path)

            # Initialize all the variables
            init_variables = tf.global_variables_initializer()
            session.run(init_variables)

            print("Loading Pre-trained Models ...")
            loader.restore(session, self.load_ckpt_file_path)
            print("Pre-trained Models are Loaded!")

            batch_iteration = 1

            summary = session.run(self.variable_summary)
            self.train_summary_writer.add_summary(summary, 0)

            for epoch in range(1, self.epochs + 1, 1):
                session.run([self.train_iterator.initializer, self.global_epochs.assign(epoch)])
                epoch_loss = 0.0
                epoch_accuracy = 0.0
                epoch_learning_rate = 0.0
                epoch_batch_iteration = 0
                epoch_time = 0.0
                epoch_preprocessing_time = 0.0
                epoch_training_time = 0.0

                batch_length = int(
                    math.floor(float(self.train_data.data_count) / float(self.batch_size * self.num_gpus)))

                while True:
                    iteration_start_time = time.time()
                    preprocessing_start_time = time.time()
                    try:
                        frame_vectors, target_vectors, masks = \
                            session.run(self.train_next_element)
                    except tf.errors.OutOfRangeError:
                        break

                    if len(frame_vectors) < self.batch_size * self.num_gpus:
                        frame_vectors = np.concatenate([frame_vectors,
                                                        np.repeat(np.expand_dims(np.zeros_like(frame_vectors[0]),
                                                                                 axis=0),
                                                                  self.batch_size * self.num_gpus - len(frame_vectors),
                                                                  axis=0)],
                                                       axis=0)
                        target_vectors = np.concatenate([target_vectors,
                                                         np.repeat(np.expand_dims(np.zeros_like(target_vectors[0]),
                                                                                  axis=0),
                                                                   self.batch_size * self.num_gpus - len(
                                                                       target_vectors),
                                                                   axis=0)],
                                                        axis=0)

                    epoch_preprocessing_time += time.time() - preprocessing_start_time

                    train_step_start_time = time.time()
                    _, loss, \
                    accuracy, \
                    predictions, \
                    current_lr = \
                        session.run(
                            [self.train_step,
                             self.model.loss,
                             self.model.accuracy,
                             self.model.predictions,
                             current_learning_rate],
                            feed_dict={self.model.frames: frame_vectors,
                                       self.model.targets: target_vectors
                                       })

                    epoch_training_time += time.time() - train_step_start_time
                    epoch_loss += loss
                    epoch_accuracy += accuracy
                    epoch_learning_rate += current_lr

                    if (batch_iteration) % self.display_term == 0:
                        predictions = np.argmax(predictions, axis=-1)
                        targets = target_vectors

                        if len(predictions) < 3:
                            show_indices = range(0, len(predictions), 1)
                            for _ in range(3 - len(predictions)):
                                show_indices.append(random.sample(range(0, len(predictions), 1), 1)[0])
                        else:
                            show_indices = random.sample(range(0, len(predictions), 1), 3)
                        show_indices.sort()

                        target_labels = \
                            [self.dataset.label_dic[str(targets[show_index])]
                             for show_index in show_indices]
                        prediction_labels = \
                            [self.dataset.label_dic[str(predictions[show_index])]
                             for show_index in show_indices]

                        print("{:<20s}: {:05d} |{:<20s}: {:03d}({:03d}/{:03d})\n" \
                              "{:<20s}: {:.9f}/{:.5f} ({:f})\n" \
                              "Expected({:03d}): {:<32s}|Prediction({:03d}): {:<32s}\n" \
                              "Expected({:03d}): {:<32s}|Prediction({:03d}): {:<32s}\n" \
                              "Expected({:03d}): {:<32s}|Prediction({:03d}): {:<32s}".format(
                            "Epochs", epoch, "Batch Iterations", batch_iteration,
                            epoch_batch_iteration + 1, batch_length,
                            "Loss", loss, accuracy, current_lr,
                            show_indices[0] + 1, target_labels[0], show_indices[0] + 1, prediction_labels[0],
                            show_indices[1] + 1, target_labels[1], show_indices[1] + 1, prediction_labels[1],
                            show_indices[2] + 1, target_labels[2], show_indices[2] + 1, prediction_labels[2]))

                    epoch_batch_iteration += 1
                    batch_iteration += 1

                    epoch_time += time.time() - iteration_start_time
                    print("Pre-processing Time: {:.5f}".format(epoch_preprocessing_time /
                                                               epoch_batch_iteration))
                    print("Training Time: {:.5f}".format(epoch_training_time /
                                                         epoch_batch_iteration))
                    print("One-Iteration Time: {:.5f}".format(epoch_time /
                                                              epoch_batch_iteration))

                epoch_loss /= float(epoch_batch_iteration)
                epoch_accuracy /= float(epoch_batch_iteration)
                epoch_learning_rate /= float(epoch_batch_iteration)
                epoch_training_time /= float(epoch_batch_iteration)
                epoch_preprocessing_time /= float(epoch_batch_iteration)

                train_summary = session.run(self.train_summaries,
                                            feed_dict={self.loss_summary_ph: epoch_loss,
                                                       self.accuracy_summary_ph: epoch_accuracy,
                                                       self.current_learning_rate_ph: epoch_learning_rate
                                                       })
                self.train_summary_writer.add_summary(train_summary, epoch)

                print("=" * 90)
                print("Epoch {:05d} Done ... Current Batch Iterations {:07d}".format(epoch, batch_iteration))
                print("Epoch {:05d} Loss {:.8f}".format(epoch, epoch_loss))
                print("Epoch {:05d} Takes {:03d} Batch Iterations".format(epoch, epoch_batch_iteration))
                print("Epoch {:05d} Takes {:.2f} Hours".format(epoch, epoch_time / 3600.0))
                print("Epoch {:05d} Average One Loop Time {:.2f} Seconds".format(epoch,
                                                                                 epoch_time / float(
                                                                                     epoch_batch_iteration)))
                print("Epoch {:05d} Average One Preprocessing Time {:.2f} Seconds".format(epoch,
                                                                                          epoch_preprocessing_time))
                print("Epoch {:05d} Average One Train Step Time {:.2f} Seconds".format(epoch,
                                                                                       epoch_training_time))
                print("Epoch {:05d} Current Learning Rate {:f}".format(epoch, epoch_learning_rate))
                print("=" * 90)

                if (epoch) % self.validation_term == 0 or epoch == 1:
                    print("Validation on Epochs {:05d}".format(epoch))

                    validation_loss = 0.0
                    validation_accuracy = 0.0

                    loop_rounds = max(int(math.ceil(float(self.validation_size) /
                                                    float(self.validation_batch_size * self.num_gpus))),
                                      1)

                    for validation_batch_index in range(loop_rounds):
                        try:
                            frame_vectors, target_vectors, masks, identities = \
                                session.run(self.validation_next_element)
                        except tf.errors.OutOfRangeError:
                            break

                        loss, accuracy, predictions = \
                            session.run(
                                [self.model_validation.loss,
                                 self.model_validation.accuracy,
                                 self.model_validation.predictions],
                                feed_dict={self.model_validation.frames: frame_vectors,
                                           self.model_validation.targets: target_vectors
                                           })

                        validation_loss += loss
                        validation_accuracy += accuracy

                        if (validation_batch_index + 1) % self.validation_display_term == 0:
                            predictions = np.argmax(predictions, axis=-1)
                            targets = target_vectors

                            if len(predictions) < 3:
                                show_indices = range(0, len(predictions), 1)
                                for _ in range(3 - len(predictions)):
                                    show_indices.append(random.sample(range(0, len(predictions), 1), 1)[0])
                            else:
                                show_indices = random.sample(range(0, len(predictions), 1), 3)
                            show_indices.sort()

                            target_labels = \
                                [self.dataset.label_dic[str(targets[show_index])]
                                 for show_index in show_indices]
                            prediction_labels = \
                                [self.dataset.label_dic[str(predictions[show_index])]
                                 for show_index in show_indices]

                            print(
                                "{:<20s}: {:05d} |{:<20s}: {:03d}/{:03d}\n" \
                                "{:<20s}: {:.9f}/{:.5f} ({})\n" \
                                "Expected({:03d}): {:<32s}|Prediction({:03d}): {:<32s}\n" \
                                "Expected({:03d}): {:<32s}|Prediction({:03d}): {:<32s}\n" \
                                "Expected({:03d}): {:<32s}|Prediction({:03d}): {:<32s}".format(
                                    "Epochs", epoch, "Batch Iterations",
                                    validation_batch_index + 1, loop_rounds,
                                    "Loss", loss, accuracy,
                                    "VALIDATION",
                                    show_indices[0] + 1, target_labels[0],
                                    show_indices[0] + 1, prediction_labels[0],
                                    show_indices[1] + 1, target_labels[1],
                                    show_indices[1] + 1, prediction_labels[1],
                                    show_indices[2] + 1, target_labels[2],
                                    show_indices[2] + 1, prediction_labels[2]))

                    validation_loss /= float(loop_rounds)
                    validation_accuracy /= float(loop_rounds)

                    validation_summary = \
                        session.run(self.validation_summaries,
                                    feed_dict={self.loss_summary_ph: validation_loss,
                                               self.accuracy_summary_ph: validation_accuracy})
                    self.validation_summary_writer.add_summary(validation_summary, epoch)

                    validation_quality = 0.5 * validation_accuracy - 0.5 * validation_loss

                    if epoch % self.ckpt_save_term == 0:
                        # if self.previous_best_epoch and self.previous_best_epoch != epoch - self.ckpt_save_term:
                        #     weight_files = glob.glob(os.path.join(self.save_ckpt_file_folder,
                        #                                           "weights.ckpt-{}.*".format(
                        #                                               epoch - self.ckpt_save_term)))
                        #     for file in weight_files:
                        #         try:
                        #             os.remove(file)
                        #         except OSError:
                        #             pass

                        saver.save(session, os.path.join(self.save_ckpt_file_folder, "weights.ckpt"),
                                   global_step=epoch)

                    if validation_quality >= self.best_validation:
                        self.best_validation = validation_quality
                        if self.previous_best_epoch and self.previous_best_epoch % self.ckpt_save_term != 0:
                            weight_files = glob.glob(os.path.join(self.save_ckpt_file_folder,
                                                                  "weights.ckpt-{}.*".format(self.previous_best_epoch)))
                            for file in weight_files:
                                try:
                                    os.remove(file)
                                except OSError:
                                    pass

                        if epoch % self.ckpt_save_term != 0:
                            saver.save(session, os.path.join(self.save_ckpt_file_folder, "weights.ckpt"),
                                       global_step=epoch)
                        self.previous_best_epoch = epoch

                    print("Validation Results ...")
                    print("Validation Loss {:.5f}".format(validation_loss))
                    print("Validation Accuracy {:.5f}".format(validation_accuracy))
                    print("=" * 90)

    def test(self):
        print("=" * 90)
        print("Testing")
        print("=" * 90)

        self.is_server = True
        self.batch_size = 4
        self.num_gpus = 2
        self.num_workers = 20
        self.data_type = "images"
        self.dataset_name = "ucf101"
        self.dataset_split = "split01"
        self.flow_type = "tvl1"
        self.optimizer_type = "SGD"
        self.temporal_width = 250
        self.dtype = tf.float32
        self.dformat = "NCDHW"

        if self.data_type == "images":
            self.model_name = "UCF_RGB"
        elif self.data_type == "flows":
            self.model_name = "UCF_Flow"
        self.train_date = "1112"

        self.validation_batch_size = self.batch_size
        self.validation_temporal_width = self.temporal_width
        self.validation_display_term = 1

        self.dataset = self.Dataset(self)

        self.validation_data = self.dataset.getDataset("test")
        self.validation_iterator = self.validation_data.tf_dataset.make_initializable_iterator()
        self.validation_next_element = self.validation_iterator.get_next()

        self.load_ckpt_file_path = \
            os.path.join(self.dataset.root_path,
                         "networks", "weights",
                         "restore",
                         "{}_{}_{}_{}".format(self.model_name, self.dataset_name,
                                              self.dataset_split, self.train_date),
                         "weights.ckpt-{}".format(16))

        self.i3d = self.I3D(self, is_training=False, data_type=self.data_type)
        self.i3d.build_model()

        os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(device_id) for device_id in range(self.num_gpus)])
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"

        loader = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        with tf.Session() as session:
            session.run(self.validation_iterator.initializer)

            print("Loading Pre-trained Models ...")
            loader.restore(session, self.load_ckpt_file_path)
            print("Pre-trained Models are Loaded!")

            validation_loss = 0.0
            validation_accuracy = 0.0
            validation_batch_index = 0
            validation_accuracy_count = 0.0

            loop_rounds = int(
                math.floor(float(self.validation_data.data_count) / float(self.batch_size * self.num_gpus)))

            while True:
                try:
                    frame_vectors, target_vectors, identities = session.run(self.validation_next_element)
                except tf.errors.OutOfRangeError:
                    break

                valid_length = len(frame_vectors)
                if valid_length < self.batch_size * self.num_gpus:
                    frame_vectors = \
                        np.concatenate([frame_vectors,
                                        np.repeat(np.zeros_like(frame_vectors[0:1]),
                                                  self.batch_size * self.num_gpus - valid_length, axis=0)],
                                       axis=0)
                    target_vectors = \
                        np.concatenate([target_vectors,
                                        np.repeat(np.zeros_like(target_vectors[0:1]),
                                                self.batch_size * self.num_gpus - valid_length, axis=0)],
                                       axis=0)

                loss, accuracy, predictions = \
                    session.run(
                        [self.i3d.loss,
                         self.i3d.accuracy,
                         self.i3d.predictions],
                        feed_dict={self.i3d.frames: frame_vectors,
                                   self.i3d.targets: target_vectors})

                predictions = predictions[:valid_length]
                target_vectors = target_vectors[:valid_length]
                accuracy = \
                    np.sum(np.array(np.equal(np.argmax(predictions, axis=-1), target_vectors),
                                     dtype=np.float32))

                validation_loss += loss
                validation_accuracy += accuracy
                validation_accuracy_count += valid_length

                if (validation_batch_index + 1) % self.validation_display_term == 0:
                    predictions = np.argmax(predictions, axis=-1)
                    targets = target_vectors

                    if len(predictions) < 3:
                        show_indices = list(range(0, len(predictions), 1))
                        for _ in range(3 - len(predictions)):
                            show_indices.append(random.sample(range(0, len(predictions), 1), 1)[0])
                    else:
                        show_indices = random.sample(range(0, len(predictions), 1), 3)
                    show_indices.sort()

                    target_labels = \
                        [self.dataset.label_dic[str(targets[show_index])]
                         for show_index in show_indices]
                    prediction_labels = \
                        [self.dataset.label_dic[str(predictions[show_index])]
                         for show_index in show_indices]

                    print(
                        "{:<20s}: {:05d} |{:<20s}: {:03d}/{:03d}\n" \
                        "{:<20s}: {:.9f}/{:.5f} ({})\n" \
                        "Expected({:03d}): {:<32s}|Prediction({:03d}): {:<32s}\n" \
                        "Expected({:03d}): {:<32s}|Prediction({:03d}): {:<32s}\n" \
                        "Expected({:03d}): {:<32s}|Prediction({:03d}): {:<32s}".format(
                            "Epochs", 1, "Batch Iterations",
                            validation_batch_index + 1, loop_rounds,
                            "Loss", loss, accuracy / valid_length,
                            "VALIDATION",
                            show_indices[0] + 1, target_labels[0],
                            show_indices[0] + 1, prediction_labels[0],
                            show_indices[1] + 1, target_labels[1],
                            show_indices[1] + 1, prediction_labels[1],
                            show_indices[2] + 1, target_labels[2],
                            show_indices[2] + 1, prediction_labels[2]))
                validation_batch_index += 1

            validation_loss /= float(loop_rounds)
            validation_accuracy /= validation_accuracy_count

            print("Validation Results ...")
            print("Validation Loss {:.5f}".format(validation_loss))
            print("Validation Accuracy {:.5f}".format(validation_accuracy))
            print("=" * 90)

    def IS(self):
        print("=" * 90)
        print("Inception Score")
        print("=" * 90)

        self.is_server = True
        self.batch_size = 1 if self.is_server else 2
        self.num_gpus = 1 if self.is_server else 1
        self.num_workers = self.num_gpus * 20
        self.data_type = "images"
        self.dataset_name = "ucf101"
        self.dataset_split = "split01"
        self.flow_type = "tvl1"
        self.temporal_width = 16
        self.dtype = tf.float32
        self.dformat = "NCDHW"

        if self.data_type == "images":
            self.model_name = "UCF_RGB"
        elif self.data_type == "flows":
            self.model_name = "UCF_Flow"
        self.train_date = "1110"

        self.validation_batch_size = self.batch_size
        self.validation_temporal_width = self.temporal_width

        self.dataset = self.Dataset(self)

        self.num_iterations = 5

        # root_dir = os.path.join("/mnt/ssd0/damien/UCF101/frames")
        # paths = glob.glob(os.path.join(root_dir, "*"))
        # random.shuffle(paths)

        prefix_path = "/mnt/ssd0/damien/UCF101/samples"
        models = [
            "moco_contrastive_single_samples",
            "moco_rot_img_samples",
            "moco_rot_img_vid_samples",
            "moco_vid_vcop_samples"
        ]

        # root_dir = os.path.join("/mnt/ssd0/damien/UCF101/samples/moco_baseline")
        # root_dir = os.path.join("/mnt/ssd0/damien/UCF101/samples/moco_mine_full")
        # root_dir = os.path.join("/mnt/ssd0/damien/UCF101/samples/moco_img")
        # root_dir = os.path.join("/mnt/ssd0/damien/UCF101/samples/moco_vid")

        # root_dir = os.path.join("/mnt/ssd0/damien/UCF101/samples/g3_baseline")
        # root_dir = os.path.join("/mnt/ssd0/damien/UCF101/samples/g3_mine_full_v3_89000")
        # root_dir = os.path.join("/mnt/ssd0/damien/UCF101/samples/89000")

        self.load_ckpt_file_path = \
            os.path.join(self.dataset.root_path,
                         "networks", "weights",
                         "restore",
                         "{}_{}_{}_{}".format(self.model_name, self.dataset_name,
                                              self.dataset_split, self.train_date),
                         "weights.ckpt-{}".format(16))

        self.i3d = self.I3D(self, is_training=False, data_type=self.data_type)
        self.i3d.build_model()

        os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(device_id) for device_id in range(self.num_gpus)])
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"

        loader = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        inception_scores = list()
        with tf.Session() as session:
            print("Loading Pre-trained Models ...")
            loader.restore(session, self.load_ckpt_file_path)
            print("Pre-trained Models are Loaded!")

            for m_i, model in enumerate(models):
                model_inception_scores = list()
                for iter in range(self.num_iterations):
                    # paths = glob.glob(os.path.join(root_dir, str(iter + 1), "*"))
                    # paths = glob.glob(os.path.join(root_dir, "*"))
                    paths = glob.glob(os.path.join(prefix_path, model, str(iter), "*"))

                    all_predictions = list()
                    for index, path in enumerate(paths):
                        # video_frames = glob.glob(os.path.join(path, "images", "*"))[:16]
                        # video = list()
                        # for frame_path in video_frames:
                        #     this_frame = cv2.imread(frame_path)
                        #     # H, W, _ = this_frame.shape
                        #     # if H >= W:
                        #     #     ratio = H / W
                        #     #     new_W = 256
                        #     #     new_H = int(round(new_W * ratio))
                        #     # else:
                        #     #     ratio = W / H
                        #     #     new_H = 256
                        #     #     new_W = int(round(new_H * ratio))
                        #     #
                        #     # this_frame = cv2.resize(this_frame, (new_W, new_H))
                        #     new_H, new_W, _ = this_frame.shape
                        #     total_crop_height = (new_H - 224)
                        #     crop_top = total_crop_height // 2
                        #     total_crop_width = (new_W - 224)
                        #     crop_left = total_crop_width // 2
                        #     this_frame = this_frame[crop_top:crop_top + 224, crop_left:crop_left + 224, :]
                        #     video.append(this_frame)
                        # video = np.concatenate(video, axis=1)

                        video = cv2.imread(path)
                        video = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)

                        H, LW, C = video.shape
                        W = H
                        L = LW // W
                        frames = list()
                        for l_i in range(L):
                            frame = video[:, l_i * W:(l_i + 1) * W]
                            frame = cv2.resize(frame, (self.input_size[1], self.input_size[0]),
                                               interpolation=cv2.INTER_LINEAR)
                            frames.append(frame)
                        frames = np.array(frames, dtype=np.float32)
                        frames = np.divide(frames, 255.0)
                        frames = np.multiply(np.subtract(frames, 0.5), 2.0)
                        frames = np.transpose(frames, (3, 0, 1, 2))

                        predictions = \
                            session.run(
                                self.i3d.predictions,
                                feed_dict={self.i3d.frames: [frames]})

                        predictions = np.squeeze(predictions, axis=0)
                        all_predictions.append(predictions)

                        print("Model({:2d}/{:2d}): {:<32s}|Iteration ({:2d}/{:2d})|Forwarding ... {:5d}/{:5d}".format(
                            m_i + 1, len(models), model, iter + 1,
                            self.num_iterations, index + 1, len(paths)), end="\r")
                    all_predictions = np.stack(all_predictions, axis=0)

                    ys = all_predictions
                    N, C = ys.shape
                    p_all = np.mean(ys, axis=0, keepdims=True)
                    kl = np.sum(ys * np.log(ys + 1e-7) - ys * np.log(p_all + 1e-7)) / N
                    inception_score = np.exp(kl)
                    model_inception_scores.append(inception_score)
                inception_scores.append(model_inception_scores)
                print()

        for m_i, model_inception_scores in enumerate(inception_scores):
            print("=" * 90)
            print("Model({:2d}/{:2d}): {}".format(m_i + 1, len(models), models[m_i]))
            for i_i, score in enumerate(model_inception_scores):
                print("Inception Score ({}): {:.5f}".format(i_i + 1, score))
            print("Average Inception Score: {:.5f}".format(np.mean(model_inception_scores)))
        print("=" * 90)

    def kinetics_test(self):
        project_root_path = os.path.abspath("..")
        meta_folder = os.path.join(project_root_path, "meta")

        kinetics_root_path = os.path.join("/mnt/hdd0", "Kinetics", "Kinetics-400")
        kinetics_ground_truth_json_path = os.path.join(meta_folder, "kinetics-400.json")
        with open(kinetics_ground_truth_json_path, "r") as fp:
            meta_dic = json.load(fp)

        frames_folder = os.path.join(kinetics_root_path, "frames")
        validation_folder = os.path.join(kinetics_root_path, "validation")
        validation_videos = glob.glob(os.path.join(validation_folder, "*"))

        tf_data = list()
        for video_idx, validation_video in enumerate(validation_videos):
            identity = validation_video.split("/")[-1].split(".")[-2]
            label = meta_dic[identity]

            flow_count = len(glob.glob(os.path.join(frames_folder, identity, "flows_tvl1", "*")))
            if flow_count >= 500:
                tf_data.append("{} {}".format(identity, label))

            print("Checking Flows ... {:5d}/{:5d}".format(video_idx + 1, len(validation_videos)))

        def preprocess(video):
            splits = video.decode().split(" ")

            identity = splits[0]
            label = int(splits[1])

            one_frame = cv2.imread(os.path.join(frames_folder, identity, "images", "img_00001.jpg"))

            height, width, _ = one_frame.shape

            frame_length = len(glob.glob(os.path.join(frames_folder, identity, "images", "*")))

            rgb_vectors = list()
            flow_vectors = list()
            for frame_number in range(frame_length):
                image_path = os.path.join(frames_folder, identity,
                                                  "{}_{:05d}.jpg".format("images/img", frame_number + 1))
                image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                total_crop_height = (height - self.input_size[1])
                crop_top = total_crop_height // 2
                total_crop_width = (width - self.input_size[0])
                crop_left = total_crop_width // 2
                image = image[crop_top:crop_top + self.input_size[1],
                        crop_left:crop_left + self.input_size[0], :]
                image = image.astype(np.float32)
                image = np.divide(image, 255.0)
                image = np.multiply(np.subtract(image, 0.5), 2.0)

                rgb_vectors.append(image)

                flow_x_path = os.path.join(frames_folder, identity,
                                           "{}_x_{:05d}.jpg".format("flows_tvl1/flow", frame_number + 1))
                flow_x = cv2.imread(flow_x_path, cv2.IMREAD_GRAYSCALE)

                flow_y_path = os.path.join(frames_folder, identity,
                                           "{}_y_{:05d}.jpg".format("flows_tvl1/flow",
                                                                    frame_number + 1))
                flow_y = cv2.imread(flow_y_path, cv2.IMREAD_GRAYSCALE)

                flow = np.stack([flow_x, flow_y], axis=-1)

                total_crop_height = (height - self.input_size[1])
                crop_top = total_crop_height // 2
                total_crop_width = (width - self.input_size[0])
                crop_left = total_crop_width // 2
                flow = flow[crop_top:crop_top + self.input_size[1],
                       crop_left:crop_left + self.input_size[0], :]
                flow = flow.astype(np.float32)
                flow = np.divide(flow, 255.0)
                flow = np.multiply(np.subtract(flow, 0.5), 2.0)

                flow_vectors.append(flow)

            return rgb_vectors, flow_vectors, label, identity

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        validation_dataset = tf.data.Dataset.from_tensor_slices(tf_data)
        validation_dataset = validation_dataset.shuffle(buffer_size=len(tf_data))
        validation_dataset = validation_dataset.map(lambda video:
                                          tf.py_func(preprocess,
                                                     [video], [tf.float32, tf.float32, tf.int64, tf.string]),
                                          num_parallel_calls=20)
        validation_dataset = validation_dataset.batch(batch_size=1)
        validation_dataset = validation_dataset.prefetch(1)
        validation_iterator = validation_dataset.make_initializable_iterator()
        validation_next_element = validation_iterator.get_next()

        rgb_frames_ph = \
            tf.placeholder(dtype=tf.float32,
                           shape=(1,
                                  None,
                                  self.input_size[0],
                                  self.input_size[1],
                                  self.input_size[2]))

        flow_frames_ph = \
            tf.placeholder(dtype=tf.float32,
                           shape=(1,
                                  None,
                                  self.input_size[0],
                                  self.input_size[1],
                                  2))

        rgb_end_points = dict()
        i3d_rgb_net = I3D.build_model(inputs=rgb_frames_ph,
                                      weight_decay=1.0e-7,
                                      end_points=rgb_end_points,
                                      dtype=tf.float32,
                                      dformat="NDHWC",
                                      is_training=False,
                                      scope="I3D_RGB")

        end_point = "RGB_Logits"
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("AvgPool_0a_2x7x7", reuse=tf.AUTO_REUSE):
                net = tf.nn.avg_pool3d(i3d_rgb_net,
                                       [1, 2, 7, 7, 1],
                                       strides=[1, 1, 1, 1, 1],
                                       padding="VALID",
                                       data_format="NDHWC")

            with tf.variable_scope("Conv3d_0c_1x1x1", reuse=tf.AUTO_REUSE):
                kernel = tf.get_variable(name="conv_3d/kernel",
                                         dtype=tf.float32,
                                         shape=[1, 1, 1,
                                                i3d_rgb_net.get_shape()[-1],
                                                400],
                                         trainable=False)
                biases = tf.get_variable(name="conv_3d/bias",
                                         dtype=tf.float32,
                                         shape=[1, 1, 1, 1,
                                                400],
                                         trainable=False)
                conv = tf.nn.conv3d(net, kernel, [1, 1, 1, 1, 1], padding="SAME",
                                    data_format="NDHWC")
                net = tf.add(conv, biases)

            net = tf.reduce_mean(net, axis=1, keepdims=True)

            net = tf.squeeze(net, axis=[1, 2, 3])
        try:
            rgb_end_points[end_point] = \
                tf.concat([rgb_end_points[end_point], net], axis=0)
        except KeyError:
            rgb_end_points[end_point] = net

        rgb_predictions = tf.nn.softmax(net, axis=-1)

        flow_end_points = dict()
        i3d_flow_net = I3D.build_model(inputs=flow_frames_ph,
                                      weight_decay=1.0e-7,
                                      end_points=rgb_end_points,
                                      dtype=tf.float32,
                                      dformat="NDHWC",
                                      is_training=False,
                                      scope="I3D_Flow")

        end_point = "Flow_Logits"
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("AvgPool_0a_2x7x7", reuse=tf.AUTO_REUSE):
                net = tf.nn.avg_pool3d(i3d_flow_net,
                                       [1, 2, 7, 7, 1],
                                       strides=[1, 1, 1, 1, 1],
                                       padding="VALID",
                                       data_format="NDHWC")

            with tf.variable_scope("Conv3d_0c_1x1x1", reuse=tf.AUTO_REUSE):
                kernel = tf.get_variable(name="conv_3d/kernel",
                                         dtype=tf.float32,
                                         shape=[1, 1, 1,
                                                i3d_flow_net.get_shape()[-1],
                                                400],
                                         trainable=False)
                biases = tf.get_variable(name="conv_3d/bias",
                                         dtype=tf.float32,
                                         shape=[1, 1, 1, 1,
                                                400],
                                         trainable=False)
                conv = tf.nn.conv3d(net, kernel, [1, 1, 1, 1, 1], padding="SAME",
                                    data_format="NDHWC")
                net = tf.add(conv, biases)

            net = tf.reduce_mean(net, axis=1, keepdims=True)

            net = tf.squeeze(net, axis=[1, 2, 3])
        try:
            flow_end_points[end_point] = \
                tf.concat([flow_end_points[end_point], net], axis=0)
        except KeyError:
            flow_end_points[end_point] = net

        flow_predictions = tf.nn.softmax(net, axis=-1)

        rgb_logit_w = None
        rgb_logit_b = None
        rgb_load_parameters = dict()
        flow_logit_w = None
        flow_logit_b = None
        flow_load_parameters = dict()
        for param in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            if "I3D_RGB" in param.name:
                key_name = param.name[:-2]
                rgb_load_parameters[key_name] = param

            if "I3D_Flow" in param.name:
                key_name = param.name[:-2]
                flow_load_parameters[key_name] = param

            if "RGB_Logit" in param.name and "kernel" in param.name:
                rgb_logit_w = param

            if "RGB_Logit" in param.name and "bias" in param.name:
                rgb_logit_b = param

            if "Flow_Logit" in param.name and "kernel" in param.name:
                flow_logit_w = param

            if "Flow_Logit" in param.name and "bias" in param.name:
                flow_logit_b = param

        rgb_cnn_ckpt_path = os.path.join(project_root_path, "cnn", "I3D", "rgb", "model.ckpt")
        rgb_logit_npz_path = os.path.join(project_root_path, "cnn", "I3D", "pretrained", "rgb_imagenet", "model.npz")

        flow_cnn_ckpt_path = os.path.join(project_root_path, "cnn", "I3D", "flow", "model.ckpt")
        flow_logit_npz_path = os.path.join(project_root_path, "cnn", "I3D", "pretrained", "flow_imagenet", "model.npz")

        rgb_cnn_loader = tf.train.Saver(var_list=rgb_load_parameters)
        flow_cnn_loader = tf.train.Saver(var_list=flow_load_parameters)

        rgb_accuracy = 0.0
        flow_accuracy = 0.0
        num_instances = 0

        with tf.Session() as session:
            print("Loading Pre-trained Models ...")
            rgb_cnn_loader.restore(session, rgb_cnn_ckpt_path)
            flow_cnn_loader.restore(session, flow_cnn_ckpt_path)
            rgb_logit_params = np.load(rgb_logit_npz_path)
            session.run(tf.assign(rgb_logit_w, rgb_logit_params["Logits/Conv3d_0c_1x1/conv_3d/w"]))
            session.run(tf.assign(rgb_logit_b, np.reshape(rgb_logit_params["Logits/Conv3d_0c_1x1/conv_3d/b"],
                                                          (1, 1, 1, 1, 400))))
            rgb_logit_params.close()
            flow_logit_params = np.load(flow_logit_npz_path)
            session.run(tf.assign(flow_logit_w, flow_logit_params["Logits/Conv3d_0c_1x1/conv_3d/w"]))
            session.run(tf.assign(flow_logit_b, np.reshape(flow_logit_params["Logits/Conv3d_0c_1x1/conv_3d/b"],
                                                           (1, 1, 1, 1, 400))))
            flow_logit_params.close()
            print("Pre-trained Models are Loaded!")

            session.run(validation_iterator.initializer)

            while True:
                try:
                    rgb_vectors, flow_vectors, targets, identities = \
                        session.run(validation_next_element)
                except tf.errors.OutOfRangeError:
                    break

                this_rgb_predictions, this_flow_predictions = \
                    session.run([rgb_predictions, flow_predictions],
                                feed_dict={rgb_frames_ph: rgb_vectors,
                                           flow_frames_ph: flow_vectors})

                rgb_accuracy += float(np.argmax(this_rgb_predictions[0], axis=-1) == targets[0])
                flow_accuracy += float(np.argmax(this_flow_predictions[0], axis=-1) == targets[0])
                num_instances += 1
                print("Kinetics Testing ... {:5d}/{:5d}".format(num_instances, len(tf_data)))

            print("=" * 90)
            print("RGB Accuracy: {:.5f}".format(rgb_accuracy / float(num_instances)))
            print("Flow Accuracy: {:.5f}".format(flow_accuracy / float(num_instances)))
            print("=" * 90)

    def make_dataset_json(self):
        data_root_dir = os.path.join("/mnt/ssd0/damien", "UCF101")
        test_split_1_txt_path = os.path.join(data_root_dir, "ucfTrainTestlist", "testlist01.txt")
        test_split_2_txt_path = os.path.join(data_root_dir, "ucfTrainTestlist", "testlist02.txt")
        test_split_3_txt_path = os.path.join(data_root_dir, "ucfTrainTestlist", "testlist03.txt")
        class_txt_path = os.path.join(data_root_dir, "ucf101_classes.txt")
        gt_json_path = os.path.join(data_root_dir, "ucf101.json")

        test_split_1_videos = list()
        with open(test_split_1_txt_path, "r") as fp:
            while True:
                line = fp.readline()
                if not line:
                    break
                line = line.rstrip()
                index = line.find("/")
                test_split_1_videos.append(line[index + 1:])
        print("Test Split 1 Videos: {}".format(len(test_split_1_videos)))
        print(test_split_1_videos[:10])

        test_split_2_videos = list()
        with open(test_split_2_txt_path, "r") as fp:
            while True:
                line = fp.readline()
                if not line:
                    break
                line = line.rstrip()
                index = line.find("/")
                test_split_2_videos.append(line[index + 1:])
        print("Test Split 2 Videos: {}".format(len(test_split_2_videos)))
        print(test_split_2_videos[:10])

        test_split_3_videos = list()
        with open(test_split_3_txt_path, "r") as fp:
            while True:
                line = fp.readline()
                if not line:
                    break
                line = line.rstrip()
                index = line.find("/")
                test_split_3_videos.append(line[index + 1:])
        print("Test Split 3 Videos: {}".format(len(test_split_3_videos)))
        print(test_split_3_videos[:10])

        videos = glob.glob(os.path.join(data_root_dir, "videos", "*"))
        print("All videos: {}".format(len(videos)))
        print(videos[:10])

        classes = list()
        for video in videos:
            index = os.path.basename(video)[2:].find("_")
            class_name = os.path.basename(video)[2:][:index]
            classes.append(class_name)

        classes = set(classes)
        classes = list(classes)
        classes.sort()
        # with open(class_txt_path, "w") as fp:
        #     for class_index, class_name in enumerate(classes):
        #         fp.write("{} {}\n".format(class_name, class_index))

        print("Classes: {}".format(len(classes)))
        print(classes[:10])

        gt_json = dict()
        gt_json["database"] = dict()
        for idx, video in enumerate(videos):
            index = os.path.basename(video)[2:].find("_")
            class_name = os.path.basename(video)[2:][:index]
            class_index = classes.index(class_name)
            video_identity = os.path.basename(video).split(".")[0]
            gt_json["database"][video_identity] = dict()
            gt_json["database"][video_identity]["annotations"] = class_index
            gt_json["database"][video_identity]["splits"] = {
                "split01": "training" if os.path.basename(video) not in test_split_1_videos else "validation",
                "split02": "training" if os.path.basename(video) not in test_split_2_videos else "validation",
                "split03": "training" if os.path.basename(video) not in test_split_3_videos else "validation"
            }
            # dst_path = os.path.join(data_root_dir, subset, os.path.basename(video))
            # copyfile(video, dst_path)
            print("Make data json {:5d}/{:5d}".format(idx + 1, len(videos)))

        with open(gt_json_path, "w") as fp:
            json.dump(gt_json, fp, indent=4, sort_keys=True)

    class Dataset():

        def __init__(self, networks):
            self.root_path = os.path.abspath("..")
            self.video_fps = 25.0
            # self.video_fps = -1
            self.networks = networks

            self.meta_folder = os.path.join(self.root_path, "meta")
            if self.networks.dataset_name == "ucf101":
                self.dataset_folder = os.path.join("/mnt/hdd1/UCF101")
                self.target_path = os.path.join(self.meta_folder, "ucf101.json")
                self.class_label_path = os.path.join(self.meta_folder, "ucf101_classes.txt")
            elif self.networks.dataset_name == "activitynet":
                self.dataset_folder = os.path.join("/mnt/hdd0/ActivityNet/v1.3")
                self.target_path = os.path.join(self.meta_folder, "activity_net.v1.3.min.json")
                self.class_label_path = os.path.join(self.meta_folder, "activitynet_classes.txt")

            if self.video_fps >= 25.0:
                self.frames_folder = os.path.join(self.dataset_folder, "frames")
            else:
                self.frames_folder = os.path.join(self.dataset_folder, "frames_adaptive")
            self.videos_folder = os.path.join(self.dataset_folder, "videos")

            self.category_dic = self.getCategoryDic()
            self.label_dic = self.getLabelDic()
            self.number_of_classes = len(self.category_dic)

            with open(self.target_path, "r") as fp:
                self.meta_dic = json.load(fp)

            if self.networks.data_type == "images":
                self.prefix = "images/img"
            elif self.networks.data_type == "flows":
                self.prefix = "flows_{}/flow".format(self.networks.flow_type)

        def getDataset(self, mode, dataset_type=None):
            if mode == "train":
                train = self.Train(self)
                validation = self.Validation(self)
                return train, validation
            elif mode == "test":
                test = self.Test(self)
                return test
            elif mode == "make":
                make = self.Make(self, dataset_type)
                return make

        def getCategoryDic(self):
            categories = {}
            with open(self.class_label_path, "r") as fp:
                while True:
                    line = fp.readline()
                    splits = line[:-1].split()
                    if len(splits) < 2:
                        break
                    category = splits[0]
                    class_number = int(splits[1])
                    categories[category] = class_number

            return categories

        def getLabelDic(self):
            labels = dict()
            with open(self.class_label_path, "r") as fp:
                while True:
                    line = fp.readline()
                    splits = line[:-1].split()
                    if len(splits) < 2:
                        break
                    category = splits[0]
                    class_number = splits[1]
                    labels[class_number] = category

            return labels

        class Train():

            def __init__(self, dataset):
                self.dataset = dataset

                print("Converting Json Train Data to Tensor Data ...")
                if self.dataset.networks.dataset_name == "ucf101":
                    if self.dataset.video_fps >= 25.0:
                        json_file_path = os.path.join(self.dataset.meta_folder,
                                                      "ucf101_{}_training_data.json".format(
                                                          self.dataset.networks.dataset_split))
                    else:
                        json_file_path = os.path.join(self.dataset.meta_folder,
                                                      "ucf101_{}_{}_training_data.json".format(
                                                          "adaptive",
                                                          self.dataset.networks.dataset_split))
                elif self.dataset.networks.dataset_name == "activitynet":
                    if self.dataset.video_fps >= 25.0:
                        json_file_path = os.path.join(self.dataset.meta_folder, "activitynet_train_data.json")
                    else:
                        json_file_path = os.path.join(self.dataset.meta_folder,
                                                      "activitynet_{}_train_data.json".format(
                                                          int(self.dataset.video_fps)))

                if os.path.exists(json_file_path):
                    with open(json_file_path, "r") as fp:
                        tf_data = json.load(fp)
                else:
                    print("There is no json file. Make the json file")
                    videos = glob.glob(os.path.join(self.dataset.videos_folder, "*"))
                    tf_data = list()
                    for index, video in enumerate(videos):
                        identity = os.path.basename(video).split(".")[-2]
                        frames = len(glob.glob(os.path.join(self.dataset.frames_folder, identity, "images", "*")))

                        splits = self.dataset.meta_dic["database"][identity]["splits"]
                        split = splits[str(self.dataset.networks.dataset_split)]
                        if split == "training":
                            annotations = self.dataset.meta_dic["database"][identity]["annotations"]

                            if not frames:
                                continue

                            tf_datum = "{} {} {}".format(identity, frames, annotations)
                            tf_data.append(tf_datum)

                            print("VIDEO {}: {:05d}/{:05d} Done".format(identity, index + 1, len(videos)))
                        else:
                            print("VIDEO {}: {:05d}/{:05d} Pass".format(identity, index + 1, len(videos)))

                    with open(json_file_path, "w") as fp:
                        json.dump(tf_data, fp, indent=4, sort_keys=True)

                self.data_count = len(tf_data)

                print("Making Tensorflow Train Dataset Object ... {} Instances".format(len(tf_data)))
                batch_size = self.dataset.networks.batch_size * self.dataset.networks.num_gpus
                train_dataset = tf.data.Dataset.from_tensor_slices(tf_data)
                train_dataset = train_dataset.shuffle(buffer_size=len(tf_data))
                train_dataset = train_dataset.prefetch(5 * batch_size)
                train_dataset = train_dataset.map(lambda video:
                                                  tf.py_func(self.sample,
                                                             [video], [tf.float32, tf.int64, tf.float32]),
                                                  num_parallel_calls=self.dataset.networks.num_workers)
                train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True)
                train_dataset = train_dataset.prefetch(5)
                self.tf_dataset = train_dataset

            def sample(self, video):
                splits = video.decode().split(" ")

                identity = splits[0]
                frame_length = int(splits[1])
                class_index = int(splits[2])

                target_frames = list()
                if frame_length >= self.dataset.networks.temporal_width:
                    start_index = random.choice(list(range(1,
                                                           frame_length -
                                                           self.dataset.networks.temporal_width + 1 + 1,
                                                           1)))
                    end_index = start_index + self.dataset.networks.temporal_width - 1
                    target_frames = list(range(start_index, end_index + 1, 1))
                else:
                    frame_index = 0
                    while True:
                        sampled_frame = 1 + (frame_index % frame_length)
                        target_frames.append(sampled_frame)
                        frame_index += 1
                        if frame_index >= self.dataset.networks.temporal_width:
                            break

                one_frame = cv2.imread(os.path.join(self.dataset.frames_folder, identity, "images", "img_00001.jpg"))

                height, width, _ = one_frame.shape

                total_crop_height = (height - self.dataset.networks.input_size[1])
                crop_top = int(np.random.uniform(low=0, high=total_crop_height + 1))
                total_crop_width = (width - self.dataset.networks.input_size[0])
                crop_left = int(np.random.uniform(low=0, high=total_crop_width + 1))

                is_flip = np.random.choice([True, False], 1)

                frame_vectors = list()
                for frame_index in target_frames:
                    if self.dataset.networks.data_type == "images":
                        if frame_index < 1 or frame_index > frame_length:
                            image = np.zeros(dtype=np.float32,
                                             shape=(self.dataset.networks.input_size[1],
                                                    self.dataset.networks.input_size[0],
                                                    3))
                        else:
                            image_path = os.path.join(self.dataset.frames_folder, identity,
                                                      "{}_{:05d}.jpg".format(self.dataset.prefix, frame_index))
                            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                            image = image[crop_top:crop_top + self.dataset.networks.input_size[1],
                                    crop_left:crop_left + self.dataset.networks.input_size[0], :]
                            if is_flip:
                                image = cv2.flip(image, 1)

                            image = image.astype(np.float32)
                            image = np.divide(image, 255.0)
                            image = np.multiply(np.subtract(image, 0.5), 2.0)

                        frame_vectors.append(image)
                    elif self.dataset.networks.data_type == "flows":
                        if frame_index < 1 or frame_index > frame_length:
                            flow = np.zeros(dtype=np.float32,
                                            shape=(self.dataset.networks.input_size[1],
                                                   self.dataset.networks.input_size[0],
                                                   2))
                        else:
                            flow_x_path = os.path.join(self.dataset.frames_folder, identity,
                                                       "{}_x_{:05d}.jpg".format(self.dataset.prefix, frame_index))
                            flow_x = cv2.imread(flow_x_path, cv2.IMREAD_GRAYSCALE)

                            flow_y_path = os.path.join(self.dataset.frames_folder, identity,
                                                       "{}_y_{:05d}.jpg".format(self.dataset.prefix, frame_index))
                            flow_y = cv2.imread(flow_y_path, cv2.IMREAD_GRAYSCALE)

                            flow = np.stack([flow_x, flow_y], axis=-1)

                            flow = flow[crop_top:crop_top + self.dataset.networks.input_size[1],
                                   crop_left:crop_left + self.dataset.networks.input_size[0], :]
                            if is_flip:
                                flow = cv2.flip(flow, 1)

                            flow = flow.astype(np.float32)
                            flow = np.divide(flow, 255.0)
                            flow = np.multiply(np.subtract(flow, 0.5), 2.0)

                        frame_vectors.append(flow)

                if self.dataset.networks.dformat == "NCDHW":
                    frame_vectors = np.transpose(frame_vectors, [3, 0, 1, 2])
                target = np.array(class_index, dtype=np.int64)

                masks = np.ones(dtype=np.float32, shape=(8 * 7 * 7))
                random_index = random.choice(range(8 * 7 * 7))
                masks[random_index] = 0.0

                return frame_vectors, target, masks

            def preprocessing(self, batch_datum):
                splits = tf.string_split([batch_datum], delimiter=" ").values

                image_path = tf.cast(splits[1], tf.string)
                target = tf.string_to_number(splits[2], tf.int64)
                new_width = tf.string_to_number(splits[3], tf.int32)
                new_height = tf.string_to_number(splits[4], tf.int32)
                crop_top = tf.string_to_number(splits[5], tf.int32)
                crop_left = tf.string_to_number(splits[6], tf.int32)
                is_flip = tf.cast(tf.string_to_number(splits[7], tf.int32), tf.bool)

                image = tf.read_file(image_path)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.image.resize_images(image,
                                               [new_height, new_width],
                                               method=tf.image.ResizeMethod.BILINEAR,
                                               align_corners=False)

                image = tf.slice(
                    image, [crop_top, crop_left, 0], [self.dataset.input_size[1], self.dataset.input_size[0], -1])

                image = tf.cond(is_flip, lambda: tf.image.flip_left_right(image), lambda: image)

                image.set_shape([self.dataset.input_size[1], self.dataset.input_size[0], 3])

                image = tf.divide(tf.cast(image, tf.float32), 255.0)

                frame_vector = tf.multiply(tf.subtract(image, 0.5), 2.0)

                return frame_vector, target

        class Validation():

            def __init__(self, dataset):
                self.dataset = dataset

                print("Converting Json Validation Data to Tensor Data ...")
                if self.dataset.networks.dataset_name == "ucf101":
                    if self.dataset.video_fps >= 25.0:
                        json_file_path = os.path.join(self.dataset.meta_folder,
                                                      "ucf101_{}_validation_data.json".format(
                                                          self.dataset.networks.dataset_split))
                    else:
                        json_file_path = os.path.join(self.dataset.meta_folder,
                                                      "ucf101_{}_{}_validation_data.json".format(
                                                          "adaptive",
                                                          self.dataset.networks.dataset_split))
                elif self.dataset.networks.dataset_name == "activitynet":
                    if self.dataset.video_fps >= 25.0:
                        json_file_path = os.path.join(self.dataset.meta_folder, "activitynet_validation_data.json")
                    else:
                        json_file_path = os.path.join(self.dataset.meta_folder,
                                                      "activitynet_{}_validation_data.json".format(
                                                          int(self.dataset.video_fps)))

                if os.path.exists(json_file_path):
                    with open(json_file_path, "r") as fp:
                        tf_data = json.load(fp)
                else:
                    print("There is no json file. Make the json file")
                    videos = glob.glob(os.path.join(self.dataset.videos_folder, "*"))
                    tf_data = list()
                    for index, video in enumerate(videos):
                        identity = os.path.basename(video).split(".")[-2]
                        frames = len(glob.glob(os.path.join(self.dataset.frames_folder, identity, "images", "*")))

                        splits = self.dataset.meta_dic["database"][identity]["splits"]
                        split = splits[str(self.dataset.networks.dataset_split)]
                        if split == "validation":
                            annotations = self.dataset.meta_dic["database"][identity]["annotations"]

                            if not frames:
                                continue

                            tf_datum = "{} {} {}".format(identity, frames, annotations)
                            tf_data.append(tf_datum)

                            print("VIDEO {}: {:05d}/{:05d} Done".format(identity, index + 1, len(videos)))
                        else:
                            print("VIDEO {}: {:05d}/{:05d} Pass".format(identity, index + 1, len(videos)))

                    with open(json_file_path, "w") as fp:
                        json.dump(tf_data, fp, indent=4, sort_keys=True)

                print("Making Tensorflow Validation Dataset Object ... {} Instances".format(len(tf_data)))
                self.data_count = len(tf_data)
                batch_size = self.dataset.networks.validation_batch_size * self.dataset.networks.num_gpus
                validation_dataset = tf.data.Dataset.from_tensor_slices(tf_data)
                validation_dataset = validation_dataset.shuffle(buffer_size=len(tf_data))
                validation_dataset = validation_dataset.repeat(-1)
                validation_dataset = validation_dataset.prefetch(5 * batch_size)
                validation_dataset = validation_dataset.map(lambda video:
                                                            tf.py_func(self.sample,
                                                                       [video], [tf.float32, tf.int64,
                                                                                 tf.float32, tf.string]),
                                                            num_parallel_calls=self.dataset.networks.num_workers)
                validation_dataset = validation_dataset.batch(batch_size)
                validation_dataset = validation_dataset.prefetch(5)
                self.tf_dataset = validation_dataset

            def sample(self, video):
                splits = video.decode().split(" ")

                identity = splits[0]
                frame_length = int(splits[1])
                class_index = int(splits[2])

                target_frames = list()
                if frame_length >= self.dataset.networks.validation_temporal_width:
                    start_index = random.choice(list(range(1,
                                                           frame_length -
                                                           self.dataset.networks.validation_temporal_width + 1 + 1,
                                                           1)))
                    end_index = start_index + self.dataset.networks.validation_temporal_width - 1
                    target_frames = list(range(start_index, end_index + 1, 1))
                else:
                    frame_index = 0
                    while True:
                        sampled_frame = 1 + (frame_index % frame_length)
                        target_frames.append(sampled_frame)
                        frame_index += 1
                        if frame_index >= self.dataset.networks.validation_temporal_width:
                            break

                one_frame = cv2.imread(os.path.join(self.dataset.frames_folder, identity, "images", "img_00001.jpg"))

                height, width, _ = one_frame.shape

                frame_vectors = list()
                for sampled_frame in target_frames:
                    if self.dataset.networks.data_type == "images":
                        if sampled_frame < 1 or sampled_frame > frame_length:
                            image = np.zeros(dtype=np.float32,
                                             shape=(self.dataset.networks.input_size[1],
                                                    self.dataset.networks.input_size[0],
                                                    3))
                        else:
                            image_path = os.path.join(self.dataset.frames_folder, identity,
                                                      "{}_{:05d}.jpg".format(self.dataset.prefix, sampled_frame))
                            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                            total_crop_height = (height - self.dataset.networks.input_size[1])
                            crop_top = total_crop_height // 2
                            total_crop_width = (width - self.dataset.networks.input_size[0])
                            crop_left = total_crop_width // 2
                            image = image[crop_top:crop_top + self.dataset.networks.input_size[1],
                                    crop_left:crop_left + self.dataset.networks.input_size[0], :]
                            image = image.astype(np.float32)
                            image = np.divide(image, 255.0)
                            image = np.multiply(np.subtract(image, 0.5), 2.0)

                        frame_vectors.append(image)
                    elif self.dataset.networks.data_type == "flows":
                        if sampled_frame < 1 or sampled_frame > frame_length:
                            flow = np.zeros(dtype=np.float32,
                                            shape=(self.dataset.networks.input_size[1],
                                                   self.dataset.networks.input_size[0],
                                                   2))
                        else:
                            flow_x_path = os.path.join(self.dataset.frames_folder, identity,
                                                       "{}_x_{:05d}.jpg".format(self.dataset.prefix, sampled_frame))
                            flow_x = cv2.imread(flow_x_path, cv2.IMREAD_GRAYSCALE)

                            flow_y_path = os.path.join(self.dataset.frames_folder, identity,
                                                       "{}_y_{:05d}.jpg".format(self.dataset.prefix,
                                                                                sampled_frame))
                            flow_y = cv2.imread(flow_y_path, cv2.IMREAD_GRAYSCALE)

                            if flow_x is None or flow_y is None:
                                # print("No Flow {} {:05d}".format(identity, sampled_frame))

                                flow = np.zeros(dtype=np.float32,
                                                shape=(self.dataset.networks.input_size[1],
                                                       self.dataset.networks.input_size[0],
                                                       2))
                            else:
                                flow = np.stack([flow_x, flow_y], axis=-1)

                                total_crop_height = (height - self.dataset.networks.input_size[1])
                                crop_top = total_crop_height // 2
                                total_crop_width = (width - self.dataset.networks.input_size[0])
                                crop_left = total_crop_width // 2
                                flow = flow[crop_top:crop_top + self.dataset.networks.input_size[1],
                                       crop_left:crop_left + self.dataset.networks.input_size[0], :]
                                flow = flow.astype(np.float32)
                                flow = np.divide(flow, 255.0)
                                flow = np.multiply(np.subtract(flow, 0.5), 2.0)

                        frame_vectors.append(flow)

                identities = [identity] * len(frame_vectors)

                if self.dataset.networks.dformat == "NCDHW":
                    frame_vectors = np.transpose(frame_vectors, [3, 0, 1, 2])

                target = np.array(class_index, dtype=np.int64)

                masks = np.ones(dtype=np.float32, shape=(8 * 7 * 7))
                random_index = random.choice(range(8 * 7 * 7))
                masks[random_index] = 0.0

                return frame_vectors, target, masks, identities

            def preprocessing(self, batch_datum):
                splits = tf.string_split([batch_datum], delimiter=" ").values

                image_path = tf.cast(splits[1], tf.string)
                identity = tf.cast(splits[2], tf.string)
                target = tf.string_to_number(splits[3], tf.int64)
                new_width = tf.string_to_number(splits[4], tf.int32)
                new_height = tf.string_to_number(splits[5], tf.int32)

                image = tf.read_file(image_path)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.image.resize_images(image,
                                               [new_height, new_width],
                                               method=tf.image.ResizeMethod.BILINEAR,
                                               align_corners=False)

                total_crop_height = (new_height - self.dataset.networks.input_size[1])
                crop_top = total_crop_height // 2
                total_crop_width = (new_width - self.dataset.networks.input_size[0])
                crop_left = total_crop_width // 2
                image = tf.slice(image, [crop_top, crop_left, 0],
                                 [self.dataset.networks.input_size[1], self.dataset.networks.input_size[0], -1])

                image.set_shape([self.dataset.networks.input_size[1], self.dataset.networks.input_size[0], 3])

                image = tf.divide(tf.cast(image, tf.float32), 255.0)

                frame_vector = tf.multiply(tf.subtract(tf.cast(image, tf.float32), 0.5), 2.0)

                return frame_vector, target, identity

        class Test():

            def __init__(self, dataset):
                self.dataset = dataset

                print("Converting Json Validation Data to Tensor Data ...")
                if self.dataset.networks.dataset_name == "ucf101":
                    if self.dataset.video_fps >= 25.0:
                        json_file_path = os.path.join(self.dataset.meta_folder,
                                                      "ucf101_{}_validation_data.json".format(
                                                          self.dataset.networks.dataset_split))
                    else:
                        json_file_path = os.path.join(self.dataset.meta_folder,
                                                      "ucf101_{}_{}_validation_data.json".format(
                                                          int(self.dataset.video_fps),
                                                          self.dataset.networks.dataset_split))
                elif self.dataset.networks.dataset_name == "activitynet":
                    if self.dataset.video_fps >= 25.0:
                        json_file_path = os.path.join(self.dataset.meta_folder, "activitynet_validation_data.json")
                    else:
                        json_file_path = os.path.join(self.dataset.meta_folder,
                                                      "activitynet_{}_validation_data.json".format(
                                                          int(self.dataset.video_fps)))

                if os.path.exists(json_file_path):
                    with open(json_file_path, "r") as fp:
                        tf_data = json.load(fp)
                else:
                    print("There is no json file. Make the json file")
                    videos = glob.glob(os.path.join(self.dataset.videos_folder, "*"))
                    tf_data = list()
                    for index, video in enumerate(videos):
                        identity = os.path.basename(video).split(".")[-2]
                        frames = len(glob.glob(os.path.join(self.dataset.frames_folder, identity, "images", "*")))

                        splits = self.dataset.meta_dic["database"][identity]["splits"]
                        split = splits[str(self.dataset.networks.dataset_split)]
                        if split == "validation":
                            annotations = self.dataset.meta_dic["database"][identity]["annotations"]

                            if not frames:
                                continue

                            tf_datum = "{} {} {}".format(identity, frames, annotations)
                            tf_data.append(tf_datum)

                            print("VIDEO {}: {:05d}/{:05d} Done".format(identity, index + 1, len(videos)))
                        else:
                            print("VIDEO {}: {:05d}/{:05d} Pass".format(identity, index + 1, len(videos)))

                    with open(json_file_path, "w") as fp:
                        json.dump(tf_data, fp, indent=4, sort_keys=True)

                print("Making Tensorflow Validation Dataset Object ... {} Instances".format(len(tf_data)))
                self.data_count = len(tf_data)
                batch_size = self.dataset.networks.validation_batch_size * self.dataset.networks.num_gpus
                validation_dataset = tf.data.Dataset.from_tensor_slices(tf_data)
                validation_dataset = validation_dataset.prefetch(5 * batch_size)
                validation_dataset = validation_dataset.map(lambda video:
                                                            tf.py_func(self.sample,
                                                                       [video], [tf.float32, tf.int64, tf.string]),
                                                            num_parallel_calls=self.dataset.networks.num_workers)
                validation_dataset = validation_dataset.batch(batch_size)
                validation_dataset = validation_dataset.prefetch(5)
                self.tf_dataset = validation_dataset

            def sample(self, video):
                splits = video.decode().split(" ")

                identity = splits[0]
                frame_length = int(splits[1])
                class_index = int(splits[2])

                if self.dataset.networks.temporal_width > 0:
                    target_frames = list()
                    if frame_length >= self.dataset.networks.validation_temporal_width:
                        start_index = 1 + (frame_length - self.dataset.networks.validation_temporal_width) // 2
                        end_index = start_index + self.dataset.networks.validation_temporal_width - 1
                        target_frames = list(range(start_index, end_index + 1, 1))
                    else:
                        frame_index = 0
                        while True:
                            sampled_frame = 1 + (frame_index % frame_length)
                            target_frames.append(sampled_frame)
                            frame_index += 1
                            if frame_index >= self.dataset.networks.validation_temporal_width:
                                break
                else:
                    target_frames = range(1, frame_length + 1, 1)

                one_frame = cv2.imread(os.path.join(self.dataset.frames_folder, identity, "images", "img_00001.jpg"))

                height, width, _ = one_frame.shape

                frame_vectors = list()
                for sampled_frame in target_frames:
                    if self.dataset.networks.data_type == "images":
                        if sampled_frame < 1 or sampled_frame > frame_length:
                            image = np.zeros(dtype=np.float32,
                                             shape=(self.dataset.networks.input_size[1],
                                                    self.dataset.networks.input_size[0],
                                                    3))
                        else:
                            image_path = os.path.join(self.dataset.frames_folder, identity,
                                                      "{}_{:05d}.jpg".format(self.dataset.prefix, sampled_frame))
                            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                            total_crop_height = (height - self.dataset.networks.input_size[1])
                            crop_top = total_crop_height // 2
                            total_crop_width = (width - self.dataset.networks.input_size[0])
                            crop_left = total_crop_width // 2
                            image = image[crop_top:crop_top + self.dataset.networks.input_size[1],
                                    crop_left:crop_left + self.dataset.networks.input_size[0], :]
                            image = image.astype(np.float32)
                            image = np.divide(image, 255.0)
                            image = np.multiply(np.subtract(image, 0.5), 2.0)

                        frame_vectors.append(image)
                    elif self.dataset.networks.data_type == "flows":
                        if sampled_frame < 1 or sampled_frame > frame_length:
                            flow = np.zeros(dtype=np.float32,
                                            shape=(self.dataset.networks.input_size[1],
                                                   self.dataset.networks.input_size[0],
                                                   2))
                        else:
                            flow_x_path = os.path.join(self.dataset.frames_folder, identity,
                                                       "{}_x_{:05d}.jpg".format(self.dataset.prefix, sampled_frame))
                            flow_x = cv2.imread(flow_x_path, cv2.IMREAD_GRAYSCALE)

                            flow_y_path = os.path.join(self.dataset.frames_folder, identity,
                                                       "{}_y_{:05d}.jpg".format(self.dataset.prefix,
                                                                                sampled_frame))
                            flow_y = cv2.imread(flow_y_path, cv2.IMREAD_GRAYSCALE)

                            if flow_x is None or flow_y is None:
                                # print("No Flow {} {:05d}".format(identity, sampled_frame))

                                flow = np.zeros(dtype=np.float32,
                                                shape=(self.dataset.networks.input_size[1],
                                                       self.dataset.networks.input_size[0],
                                                       2))
                            else:
                                flow = np.stack([flow_x, flow_y], axis=-1)

                                total_crop_height = (height - self.dataset.networks.input_size[1])
                                crop_top = total_crop_height // 2
                                total_crop_width = (width - self.dataset.networks.input_size[0])
                                crop_left = total_crop_width // 2
                                flow = flow[crop_top:crop_top + self.dataset.networks.input_size[1],
                                       crop_left:crop_left + self.dataset.networks.input_size[0], :]
                                flow = flow.astype(np.float32)
                                flow = np.divide(flow, 255.0)
                                flow = np.multiply(np.subtract(flow, 0.5), 2.0)

                        frame_vectors.append(flow)

                identities = [identity] * len(frame_vectors)

                if self.dataset.networks.dformat == "NCDHW":
                    frame_vectors = np.transpose(frame_vectors, [3, 0, 1, 2])

                target = np.array(class_index, dtype=np.int64)

                return frame_vectors, target, identities

            def preprocessing(self, batch_datum):
                splits = tf.string_split([batch_datum], delimiter=" ").values

                image_path = tf.cast(splits[1], tf.string)
                identity = tf.cast(splits[2], tf.string)
                target = tf.string_to_number(splits[3], tf.int64)
                new_width = tf.string_to_number(splits[4], tf.int32)
                new_height = tf.string_to_number(splits[5], tf.int32)

                image = tf.read_file(image_path)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.image.resize_images(image,
                                               [new_height, new_width],
                                               method=tf.image.ResizeMethod.BILINEAR,
                                               align_corners=False)

                total_crop_height = (new_height - self.dataset.networks.input_size[1])
                crop_top = total_crop_height // 2
                total_crop_width = (new_width - self.dataset.networks.input_size[0])
                crop_left = total_crop_width // 2
                image = tf.slice(image, [crop_top, crop_left, 0],
                                 [self.dataset.networks.input_size[1], self.dataset.networks.input_size[0], -1])

                image.set_shape([self.dataset.networks.input_size[1], self.dataset.networks.input_size[0], 3])

                image = tf.divide(tf.cast(image, tf.float32), 255.0)

                frame_vector = tf.multiply(tf.subtract(tf.cast(image, tf.float32), 0.5), 2.0)

                return frame_vector, target, identity

    class AdamOptimizer(tf.train.AdamOptimizer):

        def _apply_dense(self, grad, var):
            global current_learning_rate

            beta1_power, beta2_power = self._get_beta_accumulators()
            beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
            beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
            lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
            beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
            beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
            epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
            lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

            # m_t = beta1 * m + (1 - beta1) * g_t
            m = self.get_slot(var, "m")
            m_t = m * beta1_t + (1 - beta1_t)
            # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
            v = self.get_slot(var, "v")
            v_t = v * beta2_t + (1 - beta2_t)
            v_sqrt = math_ops.sqrt(v_t)
            the_step = tf.reduce_mean(lr * m_t / (v_sqrt + epsilon_t))

            # tf.summary.histogram("learning_rate/{}".format(var.name), step)
            current_learning_rate.append(the_step)

            return super(Networks.AdamOptimizer, self)._apply_dense(grad, var)

    class Model():

        def __init__(self, networks, is_training, phase, data_type,
                     batch_size=None, device_id=None, num_classes=None):
            self.networks = networks
            self.is_training = is_training
            self.phase = phase
            self.data_type = data_type

            self.dropout_prob = 0.5
            self.weight_decay = 1.0e-7
            self.K = 200

            self.solver_gamma = 1.0

            if batch_size is None:
                self.batch_size = \
                    self.networks.batch_size if self.is_training \
                        else self.networks.validation_batch_size
            elif batch_size == 1:
                self.batch_size = None
            else:
                self.batch_size = batch_size

            if self.networks.temporal_width == -1:
                self.temporal_width = None
            else:
                self.temporal_width = self.networks.temporal_width if self.is_training \
                    else self.networks.validation_temporal_width

            self.device_id = device_id
            self.num_classes = \
                num_classes if num_classes is not None \
                    else self.networks.dataset.number_of_classes
            self.num_gpus = self.networks.num_gpus if self.device_id is None else 1
            self.name = self.networks.model_name

            if self.data_type == "images":
                self.input_size = (self.networks.input_size[0],
                                   self.networks.input_size[1],
                                   3)
                self.i3d_name = "I3D_RGB"
            else:
                self.input_size = (self.networks.input_size[0],
                                   self.networks.input_size[1],
                                   2)
                self.i3d_name = "I3D_Flow"

        def build_model(self):
            self.gradients = list()
            self.loss = 0.0
            self.accuracy = 0.0
            self.solver_loss = 0.0
            self.reconstruction_loss = 0.0
            self.predictions = list()
            self.solver_predictions = list()
            self.reconstruction_predictions = list()

            kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
            kernel_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)
            bias_initializer = tf.zeros_initializer()
            # bias_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)
            bias_regularizer = None

            batch_norm_decay = 0.9997
            batch_norm_epsilon = 0.001

            batch_size = \
                self.batch_size if self.batch_size is None \
                    else self.batch_size * self.num_gpus
            if self.networks.dformat == "NDHWC":
                self.frames = \
                    tf.placeholder(dtype=self.networks.dtype, shape=(batch_size,
                                                                     self.temporal_width,
                                                                     self.input_size[0],
                                                                     self.input_size[1],
                                                                     self.input_size[2]))
            else:
                self.frames = \
                    tf.placeholder(dtype=self.networks.dtype, shape=(batch_size,
                                                                     self.input_size[2],
                                                                     self.temporal_width,
                                                                     self.input_size[0],
                                                                     self.input_size[1]))

            if self.phase == "finetuning":
                self.targets = tf.placeholder(dtype=tf.int64,
                                              shape=(batch_size, ),
                                              name="targets")
            else:
                self.masks = tf.placeholder(dtype=tf.float32,
                                            shape=(batch_size, 8 * 7 * 7),
                                            name="masks")

            self.end_points = dict()
            for device_id in range(self.num_gpus):
                with tf.device("/gpu:{:d}".format(device_id if self.device_id is None else self.device_id)):
                    with tf.name_scope("tower_{:02d}".format(device_id)):
                        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                            if self.batch_size is not None:
                                inputs = self.frames[self.batch_size * device_id:
                                                     self.batch_size * (device_id + 1)]
                            else:
                                inputs = self.frames
                            end_point = "Encoder"
                            with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
                                net = I3D.build_model(inputs=inputs,
                                                      weight_decay=self.weight_decay,
                                                      end_points=self.end_points,
                                                      dtype=self.networks.dtype,
                                                      dformat=self.networks.dformat,
                                                      is_training=self.is_training,
                                                      scope=self.i3d_name)

                                encoder_net = tf.identity(net)

                            if self.phase == "pretraining":
                                masks = self.masks[self.batch_size * device_id:
                                                   self.batch_size * (device_id + 1)]

                                end_point = "Codebook"
                                with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
                                    codebook = \
                                        tf.get_variable(name="codebook",
                                                        dtype=self.networks.dtype,
                                                        shape=[self.K, net.get_shape()[-1]],
                                                        initializer=kernel_initializer,
                                                        regularizer=kernel_regularizer,
                                                        trainable=self.is_training)

                                end_point = "VQ"
                                with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
                                    # net: N, T, H, W, C
                                    # codebook: K, C
                                    # T_pooled = tf.reduce_mean(net, axis=(2, 3))
                                    # N, T, C = T_pooled.get_shape().as_list()
                                    # K, _ = T_codebook.get_shape().as_list()

                                    # N, K, T, H, W
                                    N, T, H, W, C = net.get_shape().as_list()
                                    K, _ = codebook.get_shape().as_list()
                                    distances = tf.reduce_sum(tf.square(tf.subtract(
                                        tf.expand_dims(net, axis=1),
                                        tf.reshape(codebook, (1, K, 1, 1, 1, C)))), axis=-1)

                                    # N, T, H, W
                                    max_indices = tf.argmax(distances, axis=1)
                                    max_indices = tf.reshape(max_indices, (-1, ))
                                    solver_targets = tf.one_hot(max_indices, self.K)
                                    solver_targets = tf.reshape(solver_targets, (N, T, H, W, K))

                                    gathered_words = tf.gather(codebook, max_indices)
                                    gathered_words = tf.reshape(gathered_words, (N, T, H, W, C))

                                    # N, T, H, W, C = net.get_shape().as_list()
                                    # K, _ = codebook.get_shape().as_list()
                                    #
                                    # solver_targets = \
                                    #     tf.matmul(net,
                                    #               tf.expand_dims(tf.transpose(codebook, (1, 0)), axis=0))
                                    # # N, T, H, W, K
                                    # solver_targets = tf.nn.softmax(solver_targets, axis=-1)
                                    #
                                    # gathered_words = tf.multiply(tf.reshape(codebook, (1, 1, 1, 1, K, C)),
                                    #                              tf.expand_dims(solver_targets, axis=-1))
                                    # # N, T, C
                                    # gathered_words = tf.reduce_sum(gathered_words, axis=-2)

                                end_point = "Solver"
                                net = tf.identity(encoder_net)
                                masks = tf.reshape(masks, (-1, 8, 7, 7))
                                net = tf.multiply(net, tf.expand_dims(masks, axis=-1))
                                with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
                                    with tf.variable_scope("PositionEmbeddings_0a", reuse=tf.AUTO_REUSE):
                                        N, T, H, W, C = net.get_shape().as_list()
                                        PEs = tf.get_variable(name="position_embeddings",
                                                              dtype=self.networks.dtype,
                                                              shape=[T, H, W, C],
                                                              initializer=kernel_initializer,
                                                              trainable=self.is_training)
                                        net = net + tf.expand_dims(PEs, axis=0)

                                    net = tf.reshape(net, (N, -1, C))

                                    with tf.variable_scope("SelfAttention_0a", reuse=tf.AUTO_REUSE):
                                        N, W, C = net.get_shape().as_list()
                                        outputs = tf.identity(net)

                                        with tf.variable_scope("Q", reuse=tf.AUTO_REUSE):
                                            kernel = tf.get_variable(name="conv_1d/kernel",
                                                                     dtype=tf.float32,
                                                                     shape=[1, C, C],
                                                                     initializer=kernel_initializer,
                                                                     regularizer=kernel_regularizer,
                                                                     trainable=self.is_training)
                                            Q = tf.nn.conv1d(outputs, kernel, [1, 1, 1], padding="SAME")

                                        with tf.variable_scope("K", reuse=tf.AUTO_REUSE):
                                            kernel = tf.get_variable(name="conv_1d/kernel",
                                                                     dtype=tf.float32,
                                                                     shape=[1, C, C],
                                                                     initializer=kernel_initializer,
                                                                     regularizer=kernel_regularizer,
                                                                     trainable=self.is_training)
                                            K = tf.nn.conv1d(outputs, kernel, [1, 1, 1], padding="SAME")

                                        with tf.variable_scope("V", reuse=tf.AUTO_REUSE):
                                            kernel = tf.get_variable(name="conv_1d/kernel",
                                                                     dtype=tf.float32,
                                                                     shape=[1, C, C],
                                                                     initializer=kernel_initializer,
                                                                     regularizer=kernel_regularizer,
                                                                     trainable=self.is_training)
                                            V = tf.nn.conv1d(outputs, kernel, [1, 1, 1], padding="SAME")

                                        # split and concat
                                        # hN x W x C/h
                                        Q = tf.concat(tf.split(Q, 8, axis=-1), axis=0)
                                        K = tf.concat(tf.split(K, 8, axis=-1), axis=0)
                                        V = tf.concat(tf.split(V, 8, axis=-1), axis=0)

                                        hN, W, c_h = Q.get_shape().as_list()

                                        # hN x C/h x W
                                        K = tf.transpose(K, (0, 2, 1))

                                        # hN x W x W
                                        outputs = tf.matmul(Q, K)
                                        outputs = tf.divide(outputs, tf.sqrt(tf.constant(c_h, dtype=tf.float32)))

                                        outputs = tf.nn.softmax(outputs, -1)
                                        scores = tf.reshape(outputs, (N, 8, W, W))

                                        # hN x W x C/h
                                        outputs = tf.layers.dropout(outputs, rate=0.1, training=self.is_training)
                                        outputs = tf.matmul(outputs, V)

                                        # N x W x C
                                        outputs = tf.concat(tf.split(outputs, 8, axis=0), axis=-1)

                                        attention_scores = scores

                                        with tf.variable_scope("FC", reuse=tf.AUTO_REUSE):
                                            kernel = tf.get_variable(name="fc/kernel",
                                                                     dtype=tf.float32,
                                                                     shape=[outputs.get_shape().as_list()[-1], C],
                                                                     initializer=kernel_initializer,
                                                                     regularizer=kernel_regularizer,
                                                                     trainable=self.is_training)
                                            outputs = tf.matmul(outputs, kernel)

                                        outputs = tf.layers.dropout(outputs, rate=0.1, training=self.is_training)

                                        net = outputs + net

                                        net = tf.contrib.layers.layer_norm(net, trainable=self.is_training)

                                    net = tf.reshape(net, (N, T, H, -1, C))

                                    # with tf.variable_scope("Conv3d_3x1x1_1a", reuse=tf.AUTO_REUSE):
                                    #     C = net.get_shape().as_list()[-1] \
                                    #         if self.networks.dformat == "NDHWC" \
                                    #         else net.get_shape().as_list()[1]
                                    #     kernel = tf.get_variable(name="conv_3d/kernel",
                                    #                              dtype=self.networks.dtype,
                                    #                              shape=[3, 1, 1, C, C],
                                    #                              initializer=kernel_initializer,
                                    #                              regularizer=kernel_regularizer,
                                    #                              trainable=self.is_training)
                                    #     net = tf.nn.conv3d(net, kernel, [1, 1, 1, 1, 1], padding="SAME")
                                    #     bias = tf.get_variable(name="conv_3d/bias",
                                    #                            dtype=self.networks.dtype,
                                    #                            shape=[C],
                                    #                            initializer=bias_initializer,
                                    #                            regularizer=bias_regularizer,
                                    #                            trainable=self.is_training)
                                    #     net = tf.nn.bias_add(net, bias)
                                    #     net = tf.layers.batch_normalization(net,
                                    #                                         axis=-1,
                                    #                                         center=True,
                                    #                                         scale=False,
                                    #                                         momentum=batch_norm_decay,
                                    #                                         epsilon=batch_norm_epsilon,
                                    #                                         training=self.is_training,
                                    #                                         trainable=self.is_training)
                                    #     net = tf.nn.relu(net)
                                    #
                                    # with tf.variable_scope("Conv3d_1x3x3_1b", reuse=tf.AUTO_REUSE):
                                    #     C = net.get_shape().as_list()[-1] \
                                    #         if self.networks.dformat == "NDHWC" \
                                    #         else net.get_shape().as_list()[1]
                                    #     kernel = tf.get_variable(name="conv_1d/kernel",
                                    #                              dtype=self.networks.dtype,
                                    #                              shape=[1, 3, 3, C, C],
                                    #                              initializer=kernel_initializer,
                                    #                              regularizer=kernel_regularizer,
                                    #                              trainable=self.is_training)
                                    #     net = tf.nn.conv3d(net, kernel, [1, 1, 1, 1, 1], padding="SAME")
                                    #     bias = tf.get_variable(name="conv_3d/bias",
                                    #                            dtype=self.networks.dtype,
                                    #                            shape=[C],
                                    #                            initializer=bias_initializer,
                                    #                            regularizer=bias_regularizer,
                                    #                            trainable=self.is_training)
                                    #     net = tf.nn.bias_add(net, bias)
                                    #     net = tf.layers.batch_normalization(net,
                                    #                                         axis=-1,
                                    #                                         center=True,
                                    #                                         scale=False,
                                    #                                         momentum=batch_norm_decay,
                                    #                                         epsilon=batch_norm_epsilon,
                                    #                                         training=self.is_training,
                                    #                                         trainable=self.is_training)
                                    #     net = tf.nn.relu(net)

                                    with tf.variable_scope("SolverLogits", reuse=tf.AUTO_REUSE):
                                        C = net.get_shape().as_list()[-1]
                                        target_C = self.K
                                        kernel = tf.get_variable(name="conv_3d/kernel",
                                                                 dtype=self.networks.dtype,
                                                                 shape=[1, 1, 1, C, target_C],
                                                                 initializer=kernel_initializer,
                                                                 regularizer=kernel_regularizer,
                                                                 trainable=self.is_training)
                                        net = tf.nn.conv3d(net, kernel, [1, 1, 1, 1, 1], padding="SAME")
                                        bias = tf.get_variable(name="conv_3d/bias",
                                                               dtype=self.networks.dtype,
                                                               shape=[target_C],
                                                               initializer=bias_initializer,
                                                               regularizer=bias_regularizer,
                                                               trainable=self.is_training)
                                        net = tf.nn.bias_add(net, bias)

                                    p = tf.nn.softmax(net, axis=-1)
                                    self.solver_predictions.append(p)
                                    t = tf.stop_gradient(solver_targets)
                                    solver_loss = -tf.reduce_mean(t * tf.log(p + 1.0e-7), axis=-1)
                                    solver_loss = tf.multiply(solver_loss, 1.0 - masks)
                                    solver_loss = tf.reduce_sum(solver_loss, axis=(1, 2, 3))
                                    solver_loss = tf.reduce_mean(solver_loss, axis=0)

                                    self.solver_loss += solver_loss

                                end_point = "Decoder"
                                net = tf.identity(gathered_words)
                                with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
                                    for level_index in range(2):
                                        with tf.variable_scope("Level_{:02d}".format(level_index + 1)):
                                            if level_index % 2 == 0:
                                                N, T, H, W, C = net.get_shape().as_list()
                                                with tf.variable_scope("T_UpSample_0a", reuse=tf.AUTO_REUSE):
                                                    net = tf.reshape(net, (N, T, H * W, C))
                                                    net = tf.image.resize_bilinear(net, size=(T * 2, H * W))

                                                net = tf.reshape(net, (N, T * 2, H, W, C))

                                                with tf.variable_scope("Conv3d_3x1x1_1a", reuse=tf.AUTO_REUSE):
                                                    C = net.get_shape().as_list()[-1] \
                                                        if self.networks.dformat == "NDHWC" \
                                                        else net.get_shape().as_list()[1]
                                                    kernel = tf.get_variable(name="conv_3d/kernel",
                                                                             dtype=self.networks.dtype,
                                                                             shape=[3, 1, 1, C, C],
                                                                             initializer=kernel_initializer,
                                                                             regularizer=kernel_regularizer,
                                                                             trainable=self.is_training)
                                                    net = tf.nn.conv3d(net, kernel, [1, 1, 1, 1, 1], padding="SAME")
                                                    bias = tf.get_variable(name="conv_3d/bias",
                                                                           dtype=self.networks.dtype,
                                                                           shape=[C],
                                                                           initializer=bias_initializer,
                                                                           regularizer=bias_regularizer,
                                                                           trainable=self.is_training)
                                                    net = tf.nn.bias_add(net, bias)
                                                    net = tf.layers.batch_normalization(net,
                                                                                        axis=-1,
                                                                                        center=True,
                                                                                        scale=False,
                                                                                        momentum=batch_norm_decay,
                                                                                        epsilon=batch_norm_epsilon,
                                                                                        training=self.is_training,
                                                                                        trainable=self.is_training)
                                                    net = tf.nn.relu(net)

                                            N, T, H, W, C = net.get_shape().as_list()
                                            with tf.variable_scope("S_UpSample_0a", reuse=tf.AUTO_REUSE):
                                                net = tf.reshape(net, (N * T, H, W, C))
                                                net = tf.image.resize_bilinear(net, size=(H * 2, W * 2))

                                            net = tf.reshape(net, (N, T, H * 2, W * 2, C))

                                            with tf.variable_scope("Conv3d_1x3x3_1b", reuse=tf.AUTO_REUSE):
                                                C = net.get_shape().as_list()[-1] \
                                                    if self.networks.dformat == "NDHWC" \
                                                    else net.get_shape().as_list()[1]
                                                kernel = tf.get_variable(name="conv_3d/kernel",
                                                                         dtype=self.networks.dtype,
                                                                         shape=[1, 3, 3, C, C],
                                                                         initializer=kernel_initializer,
                                                                         regularizer=kernel_regularizer,
                                                                         trainable=self.is_training)
                                                net = tf.nn.conv3d(net, kernel, [1, 1, 1, 1, 1], padding="SAME")
                                                bias = tf.get_variable(name="conv_3d/bias",
                                                                       dtype=self.networks.dtype,
                                                                       shape=[C],
                                                                       initializer=bias_initializer,
                                                                       regularizer=bias_regularizer,
                                                                       trainable=self.is_training)
                                                net = tf.nn.bias_add(net, bias)
                                                net = tf.layers.batch_normalization(net,
                                                                                    axis=-1,
                                                                                    center=True,
                                                                                    scale=False,
                                                                                    momentum=batch_norm_decay,
                                                                                    epsilon=batch_norm_epsilon,
                                                                                    training=self.is_training,
                                                                                    trainable=self.is_training)
                                                net = tf.nn.relu(net)

                                    low_level_features = inputs

                                    with tf.variable_scope("ReconstructionLogits", reuse=tf.AUTO_REUSE):
                                        C = net.get_shape().as_list()[-1]
                                        target_C = low_level_features.get_shape().as_list()[-1]
                                        kernel = tf.get_variable(name="conv_3d/kernel",
                                                                 dtype=self.networks.dtype,
                                                                 shape=[1, 1, 1, C, target_C],
                                                                 initializer=kernel_initializer,
                                                                 regularizer=kernel_regularizer,
                                                                 trainable=self.is_training)
                                        net = tf.nn.conv3d(net, kernel, [1, 1, 1, 1, 1], padding="SAME")
                                        bias = tf.get_variable(name="conv_3d/bias",
                                                               dtype=self.networks.dtype,
                                                               shape=[target_C],
                                                               initializer=bias_initializer,
                                                               regularizer=bias_regularizer,
                                                               trainable=self.is_training)
                                        net = tf.nn.bias_add(net, bias)

                                    _, target_T, target_H, target_W, _ = low_level_features.get_shape().as_list()
                                    N, T, H, W, C = net.get_shape().as_list()
                                    net = tf.reshape(net, (N, T, H * W, C))
                                    net = tf.image.resize_bilinear(net, size=(target_T, H * W))
                                    net = tf.reshape(net, (N * target_T, H, W, C))
                                    net = tf.image.resize_bilinear(net, size=(target_H, target_W))
                                    net = tf.reshape(net, (N, target_T, target_H, target_W, C))

                                    self.reconstruction_predictions.append(net)

                                    reconstruction_loss = tf.reduce_mean(tf.square(net - low_level_features), axis=-1)
                                    reconstruction_loss = tf.reduce_mean(reconstruction_loss, axis=(1, 2, 3))
                                    reconstruction_loss = tf.reduce_mean(reconstruction_loss, axis=0)

                                    q_loss = tf.reduce_mean(tf.square(tf.norm(
                                        tf.stop_gradient(encoder_net) - gathered_words, axis=-1)))

                                    commit_loss = tf.reduce_mean(tf.square(tf.norm(
                                        encoder_net - tf.stop_gradient(gathered_words), axis=-1)))

                                    vq_loss = reconstruction_loss + q_loss + 0.25 * commit_loss

                                    self.reconstruction_loss += vq_loss

                                    # self.reconstruction_loss += reconstruction_loss

                                loss = self.solver_gamma * solver_loss + vq_loss
                                self.loss += loss
                            else:
                                end_point = "Logits"
                                with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
                                    with tf.variable_scope("AvgPool_0a_2x7x7", reuse=tf.AUTO_REUSE):
                                        net = tf.nn.avg_pool3d(net,
                                                               [1, 2, 7, 7, 1]
                                                               if self.networks.dformat == "NDHWC"
                                                               else [1, 1, 2, 7, 7],
                                                               strides=[1, 1, 1, 1, 1],
                                                               padding="VALID",
                                                               data_format=self.networks.dformat)

                                    with tf.variable_scope("Dropout_0b", reuse=tf.AUTO_REUSE):
                                        net = tf.nn.dropout(net,
                                                            keep_prob=0.5 if self.is_training else 1.0)

                                    with tf.variable_scope("Conv3d_0c_1x1x1", reuse=tf.AUTO_REUSE):
                                        kernel = tf.get_variable(name="conv_3d/kernel",
                                                                 dtype=self.networks.dtype,
                                                                 shape=[1, 1, 1,
                                                                        net.get_shape()[-1]
                                                                        if self.networks.dformat == "NDHWC"
                                                                        else net.get_shape()[1],
                                                                        self.num_classes],
                                                                 initializer=kernel_initializer,
                                                                 regularizer=kernel_regularizer,
                                                                 trainable=self.is_training)
                                        biases = tf.get_variable(name="conv_3d/bias",
                                                                 dtype=self.networks.dtype,
                                                                 shape=[1, 1, 1, 1,
                                                                        self.num_classes]
                                                                 if self.networks.dformat == "NDHWC"
                                                                 else [1, self.num_classes,
                                                                       1, 1, 1],
                                                                 initializer=bias_initializer,
                                                                 regularizer=bias_regularizer,
                                                                 trainable=self.is_training)
                                        conv = tf.nn.conv3d(net, kernel, [1, 1, 1, 1, 1], padding="SAME",
                                                            data_format=self.networks.dformat)
                                        net = tf.add(conv, biases)

                                    net = tf.reduce_mean(net,
                                                         axis=1 if self.networks.dformat == "NDHWC" else 2,
                                                         keepdims=True)

                                    net = tf.squeeze(net,
                                                     axis=[1, 2, 3]
                                                     if self.networks.dformat == "NDHWC"
                                                     else [2, 3, 4])
                                try:
                                    self.end_points[end_point] = \
                                        tf.concat([self.end_points[end_point], net], axis=0)
                                except KeyError:
                                    self.end_points[end_point] = net

                                self.predictions.append(tf.nn.softmax(net, axis=-1))

                                if self.batch_size is not None:
                                    targets = self.targets[
                                              self.batch_size * device_id:
                                              self.batch_size * (device_id + 1)]
                                else:
                                    targets = self.targets

                                self.accuracy += \
                                    tf.reduce_mean(
                                        tf.cast(
                                            tf.equal(
                                                tf.argmax(self.predictions[-1],
                                                          axis=-1),
                                                targets),
                                            self.networks.dtype))

                                loss = \
                                    tf.reduce_mean(
                                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                                            labels=targets,
                                            logits=net
                                        )
                                    )
                                self.loss += loss

                        if self.is_training:
                            if self.phase == "pretraining":
                                # Decoder Grads
                                decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                 scope=self.name + "/Decoder")
                                decoder_grads = list(zip(tf.gradients(vq_loss, decoder_vars), decoder_vars))
                                # Encoder Grads
                                encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                 scope=self.name + "/Encoder")
                                grad_z = tf.gradients(reconstruction_loss, gathered_words)
                                # encoder_grads = list()
                                # for i, var in enumerate(encoder_vars):
                                #     encoder_grads.append(
                                #         (tf.gradients(encoder_net, var, grad_z)[0] +
                                #          0.25 * tf.gradients(commit_loss, var)[0] +
                                #          self.solver_gamma * tf.gradients(solver_loss, var)[0], var))
                                #     print("Encoder Compute Gradients ... {:2d}|{:3d}/{:3d}".format(
                                #         device_id + 1, i + 1, len(encoder_vars)))
                                # encoder_grads = [
                                #     (tf.gradients(encoder_net, var, grad_z)[0] +
                                #      0.25 * self.networks.optimizer.compute_gradients(commit_loss, var)[0] +
                                #      self.solver_gamma * self.networks.optimizer.compute_gradients(solver_loss, var)[0], var)
                                #     for var in encoder_vars]
                                encoder_grads_01 = tf.gradients(encoder_net, encoder_vars, grad_z)
                                encoder_grads_02 = tf.gradients(commit_loss, encoder_vars)
                                encoder_grads_03 = tf.gradients(solver_loss, encoder_vars)
                                encoder_grads = \
                                    list(zip([grads_01 + 0.25 * grads_02 + self.solver_gamma * grads_03
                                              for grads_01, grads_02, grads_03
                                              in zip(encoder_grads_01, encoder_grads_02, encoder_grads_03)],
                                             encoder_vars))
                                # Embedding Grads
                                embed_grads = list(zip(tf.gradients(q_loss, codebook), [codebook]))
                                # Solver Grads
                                solver_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                scope=self.name + "/Solver")
                                solver_grads = list(zip(
                                    [self.solver_gamma * grad for grad in tf.gradients(solver_loss, solver_vars)],
                                    solver_vars))
                                gradients = decoder_grads + encoder_grads + embed_grads + solver_grads
                                if device_id == 0:
                                    # self.reg_gradients = self.networks.optimizer.compute_gradients(
                                    #     tf.losses.get_regularization_loss())
                                    all_vars = decoder_vars + encoder_vars + [codebook] + solver_vars
                                    self.reg_gradients = \
                                        list(zip(tf.gradients(tf.losses.get_regularization_loss(), all_vars), all_vars))
                            else:
                                gradients = self.networks.optimizer.compute_gradients(loss)
                                if device_id == 0:
                                    self.reg_gradients = \
                                        self.networks.optimizer.compute_gradients(tf.losses.get_regularization_loss())

                            self.gradients.append(gradients)

            with tf.device("/cpu:0"):
                self.loss /= tf.constant(self.networks.num_gpus, dtype=self.networks.dtype)
                if self.phase == "pretraining":
                    self.solver_loss /= tf.constant(self.networks.num_gpus, dtype=self.networks.dtype)
                    self.reconstruction_loss /= tf.constant(self.networks.num_gpus, dtype=self.networks.dtype)
                    self.solver_predictions = tf.concat(self.solver_predictions, axis=0)
                    self.reconstruction_predictions = tf.concat(self.reconstruction_predictions, axis=0)
                else:
                    self.accuracy /= tf.constant(self.networks.num_gpus, dtype=self.networks.dtype)
                    self.predictions = tf.concat(self.predictions, axis=0)

                if self.is_training:
                    self.average_grads = list()
                    index = 0
                    for grad_and_vars in zip(*self.gradients):
                        grads = list()
                        for g, _ in grad_and_vars:
                            expanded_g = tf.expand_dims(g, 0)
                            grads.append(expanded_g)

                        grad = tf.concat(axis=0, values=grads)
                        grad = tf.reduce_mean(grad, 0)
                        v = grad_and_vars[0][1]

                        if self.reg_gradients[index][0] is not None:
                            grad += self.reg_gradients[index][0]
                        index += 1

                        grad_and_var = (grad, v)
                        self.average_grads.append(grad_and_var)


if __name__ == "__main__":

    networks = Networks()

    networks.finetune()
