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
from shutil import rmtree
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.manifold import TSNE
import itertools


class Networks:

    def __init__(self):
        self.input_size = (224, 224, 3)

    def train(self):
        print("=" * 90)
        print("Networks Training")
        print("=" * 90)

        self.is_server = True
        self.batch_size = 16 if self.is_server else 2
        self.num_gpus = 2 if self.is_server else 1
        self.num_workers = self.num_gpus * 12
        self.data_type = "flows"
        self.dataset_name = "activitynet"
        self.flow_type = "tvl1"
        self.optimizer_type = "Adam"
        if self.dataset_name == "thumos14":
            if self.optimizer_type == "Adam":
                self.epochs = 200 if self.data_type == "images" else 300
            else:
                self.epochs = 300 if self.data_type == "images" else 450
        else:
            if self.optimizer_type == "Adam":
                self.epochs = 50 if self.data_type == "images" else 80
            else:
                self.epochs = 80 if self.data_type == "images" else 120
        self.temporal_width = 64
        self.display_term = 1
        self.dtype = tf.float32
        self.dformat = "NCDHW"

        if self.data_type == "images":
            self.model_name = "SpottingNetwork_RGB"
        elif self.data_type == "flows":
            self.model_name = "SpottingNetwork_Flow"
        now = time.localtime()
        self.train_date = "{:02d}{:02d}".format(now.tm_mon, now.tm_mday)

        self.validation_batch_size = self.batch_size
        self.validation_term = 1 if self.dataset_name == "thumos14" else 10
        self.validation_temporal_width = self.temporal_width
        self.validation_display_term = self.display_term
        self.ckpt_save_term = 10

        self.dataset = self.Dataset(self)

        self.train_data, self.validation_data = self.dataset.getDataset("train")
        self.train_iterator = self.train_data.tf_dataset.make_initializable_iterator()
        self.train_next_element = self.train_iterator.get_next()

        self.validation_iterator = self.validation_data.tf_dataset.make_one_shot_iterator()
        self.validation_next_element = self.validation_iterator.get_next()
        self.validation_size = self.validation_data.data_count // (1 if self.dataset_name == "activitynet" else 1)

        self.save_ckpt_file_folder = os.path.join(self.dataset.root_path,
                                                  "networks", "weights",
                                                  "save", "{}_{}_{}".format(self.model_name,
                                                                            self.dataset_name,
                                                                            self.train_date))
        if self.data_type == "images":
            self.load_ckpt_file_path = os.path.join(self.dataset.root_path, "cnn/I3D/rgb", "model.ckpt")
        elif self.data_type == "flows":
            self.load_ckpt_file_path = os.path.join(self.dataset.root_path, "cnn/I3D/flow", "model.ckpt")
        else:
            self.load_ckpt_file_path = os.path.join(self.dataset.root_path, "cnn/I3D/rgb", "model.ckpt")
        self.summary_folder = os.path.join(self.dataset.root_path,
                                           "networks", "summaries",
                                           "{}_{}_{}".format(self.model_name, self.dataset_name,
                                                             self.train_date))
        self.train_summary_file_path = os.path.join(self.summary_folder, "train_summary")
        self.validation_summary_file_path = os.path.join(self.summary_folder, "validation_summary")

        self.global_step = tf.Variable(0, trainable=False)
        self.global_epochs = tf.Variable(1, trainable=False)
        if self.optimizer_type == "Adam":
            self.starter_learning_rate = 2.0e-4
        else:
            self.starter_learning_rate = 1.0e-3

        if self.data_type == "images":
            if self.dataset_name == "thumos14":
                if self.optimizer_type == "Adam":
                    boundaries = [70, 150]
                else:
                    boundaries = [120, 250]
            else:
                if self.optimizer_type == "Adam":
                    boundaries = [20, 35]
                else:
                    boundaries = [40, 65]
            values = [self.starter_learning_rate,
                      self.starter_learning_rate * 1.0e-1,
                      self.starter_learning_rate * 1.0e-2]
        else:
            if self.dataset_name == "thumos14":
                if self.optimizer_type == "Adam":
                    boundaries = [70, 150, 200, 250, 290]
                else:
                    boundaries = [120, 250, 300, 370, 420]
            else:
                if self.optimizer_type == "Adam":
                    boundaries = [20, 35, 50, 60, 70]
                else:
                    boundaries = [40, 65, 80, 100, 110]
            values = [self.starter_learning_rate,
                      self.starter_learning_rate * 1.0e-1,
                      self.starter_learning_rate * 1.0e-2,
                      self.starter_learning_rate * 1.0e-1,
                      self.starter_learning_rate * 1.0e-2,
                      self.starter_learning_rate * 1.0e-3]
        self.learning_rate = tf.train.piecewise_constant(self.global_epochs, boundaries, values)

        global current_learning_rate
        current_learning_rate = list()

        if self.optimizer_type == "Adam":
            self.optimizer = self.AdamOptimizer(learning_rate=self.learning_rate)
        else:
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                        momentum=0.9)

        self.spotting_network = self.SpottingNetwork(self, is_training=True, data_type=self.data_type)
        self.spotting_network_validation = self.SpottingNetwork(self, is_training=False, data_type=self.data_type)
        self.spotting_network.build_model()
        self.spotting_network_validation.build_model()

        self.parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        load_parameters = dict()
        for param in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            if "I3D" in param.name:
                key_name = param.name.replace(self.model_name + "/", "")[:-2]
                load_parameters[key_name] = param

        self.parameter_dict = dict()
        for parameter in self.parameters:
            self.parameter_dict[parameter.name] = parameter

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            with tf.device("/cpu:0"):
                self.train_step = self.optimizer.apply_gradients(self.spotting_network.average_grads,
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
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES),
                               max_to_keep=self.epochs)

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
                        frame_vectors, target_vectors = \
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
                             self.spotting_network.loss,
                             self.spotting_network.accuracy,
                             self.spotting_network.predictions,
                             current_learning_rate],
                            feed_dict={self.spotting_network.frames: frame_vectors,
                                       self.spotting_network.targets: target_vectors
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
                            [self.dataset.label_dic[str(targets[show_index] + 1)]
                             for show_index in show_indices]
                        prediction_labels = \
                            [self.dataset.label_dic[str(predictions[show_index] + 1)]
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
                            frame_vectors, target_vectors, identities = session.run(self.validation_next_element)
                        except tf.errors.OutOfRangeError:
                            break

                        loss, accuracy, predictions = \
                            session.run(
                                [self.spotting_network_validation.loss,
                                 self.spotting_network_validation.accuracy,
                                 self.spotting_network_validation.predictions],
                                feed_dict={self.spotting_network_validation.frames: frame_vectors,
                                           self.spotting_network_validation.targets: target_vectors
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
                                [self.dataset.label_dic[str(targets[show_index] + 1)]
                                 for show_index in show_indices]
                            prediction_labels = \
                                [self.dataset.label_dic[str(predictions[show_index] + 1)]
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
                                               self.accuracy_summary_ph: validation_accuracy
                                               })
                    self.validation_summary_writer.add_summary(validation_summary, epoch)

                    validation_quality = 0.5 * validation_accuracy - 0.5 * validation_loss

                    if epoch % self.ckpt_save_term == 0:
                        if self.previous_best_epoch and self.previous_best_epoch != epoch - self.ckpt_save_term:
                            weight_files = glob.glob(os.path.join(self.save_ckpt_file_folder,
                                                                  "weights.ckpt-{}.*".format(
                                                                      epoch - self.ckpt_save_term)))
                            for file in weight_files:
                                try:
                                    os.remove(file)
                                except OSError:
                                    pass

                        saver.save(session, os.path.join(self.save_ckpt_file_folder, "weights.ckpt"),
                                   global_step=epoch)

                    if validation_quality >= self.best_validation:
                        self.best_validation = validation_quality
                        if self.previous_best_epoch:
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

    def make(self):
        print("=" * 90)
        print("Networks Making")
        print("=" * 90)

        self.is_server = True
        self.num_workers = 24 if self.is_server else 8
        self.num_gpus = 2 if self.is_server else 1
        self.dataset_name = "activitynet"
        self.make_type = "Fusion"

        self.num_crops = 5 if self.dataset_name == "thumos14" else 3
        self.batch_size = 1
        self.max_batch_size = 64
        self.temporal_width = 64
        self.validation_temporal_width = self.temporal_width
        self.step_size = 8
        # self.step_size = 16

        self.dtype = tf.float32
        self.dformat = "NCDHW"
        self.data_type = "images"
        self.flow_type = "tvl1"
        self.dataset = self.Dataset(self)
        self.dataset_type = "testing"
        self.make_data = self.dataset.getDataset("make", self.dataset_type)

        self.make_data.tf_dataset = self.make_data.tf_dataset.batch(1)
        self.make_data.tf_dataset = self.make_data.tf_dataset.prefetch(5)
        self.make_iterator = self.make_data.tf_dataset.make_initializable_iterator()
        self.make_next_element = self.make_iterator.get_next()

        if self.make_type in ["RGB", "Fusion"]:
            self.load_rgb_ckpt_file_path = \
                os.path.join(self.dataset.root_path, "networks", "weights", "restore",
                             "SpottingNetwork_RGB_activitynet_0423", "weights.ckpt-50")

        if self.make_type in ["Flow", "Fusion"]:
            self.load_flow_ckpt_file_path = \
                os.path.join(self.dataset.root_path, "networks", "weights", "restore",
                             "SpottingNetwork_Flow_activitynet_0427", "weights.ckpt-80")

        if self.make_type == "Fusion":
            self.data_folder = \
                os.path.join(self.dataset.dataset_folder,
                             "Finetuned_Data_{}_{}".format(self.dataset_name,
                                                           "0427_T16"))
        else:
            self.data_folder = \
                os.path.join(self.dataset.dataset_folder,
                             "Finetuned_Data_{}_{}".format(self.dataset_name,
                                                           "0427_T16_{}".format(self.make_type)))

        self.feature_folder = os.path.join(self.data_folder, "features")
        self.visualization_folder = os.path.join(self.data_folder, "visualization")

        folders = [self.data_folder,
                   self.feature_folder,
                   self.visualization_folder]

        for folder in folders:
            if not os.path.exists(folder):
                try:
                    os.mkdir(folder)
                except OSError:
                    pass

        if self.num_gpus >= 2:
            if self.make_type == "Fusion":
                device_ids = [0, 1]
            else:
                device_ids = [0, 0]
        else:
            device_ids = [0, 0]

        batch_size = -1 if self.batch_size > self.max_batch_size else self.batch_size
        if self.make_type in ["RGB", "Fusion"]:
            self.model_name = "SpottingNetwork_RGB"
            self.spotting_network_rgb = self.SpottingNetwork(self, is_training=False,
                                                             data_type="images",
                                                             batch_size=batch_size,
                                                             device_id=device_ids[0])
            self.spotting_network_rgb.build_model()

        if self.make_type in ["Flow", "Fusion"]:
            self.model_name = "SpottingNetwork_Flow"
            self.spotting_network_flow = self.SpottingNetwork(self, is_training=False,
                                                              data_type="flows",
                                                              batch_size=batch_size,
                                                              device_id=device_ids[1])
            self.spotting_network_flow.build_model()

        os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(device_id) for device_id in range(self.num_gpus)])
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"

        if self.make_type in ["RGB", "Fusion"]:
            rgb_loader = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                   scope="SpottingNetwork_RGB"))

        if self.make_type in ["Flow", "Fusion"]:
            flow_loader = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                   scope="SpottingNetwork_Flow"))

        cmap = plt.get_cmap("terrain")
        cNorm = matplotlib.colors.Normalize(vmin=0, vmax=self.dataset.number_of_classes - 1)
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)

        self.color_maps = scalarMap

        with tf.Session() as session:
            print("Loading Pre-trained Models ...")
            if self.make_type in ["RGB", "Fusion"]:
                rgb_loader.restore(session, self.load_rgb_ckpt_file_path)

            if self.make_type in ["Flow", "Fusion"]:
                flow_loader.restore(session, self.load_flow_ckpt_file_path)
            print("Pre-trained Models are Loaded!")

            make_batch_index = 0
            session.run(self.make_iterator.initializer)

            additional_loop = self.temporal_width // self.step_size - 1

            while True:
                try:
                    identities, first_frames, \
                    first_targets, crop_settings, \
                    segments = session.run(self.make_next_element)
                except tf.errors.OutOfRangeError:
                    break

                identity = str(identities[0].decode())

                video_feature_folder = os.path.join(self.feature_folder, identity)
                if not os.path.exists(video_feature_folder):
                    try:
                        os.mkdir(video_feature_folder)
                    except OSError:
                        pass

                this_feature_paths = list()
                if self.dataset_type == "training":
                    crop_length = self.num_crops * self.num_crops * 2
                    for idx in range(crop_length):
                        is_flipped = int(idx % 2 == 1)
                        this_feature_path = \
                            os.path.join(video_feature_folder,
                                         "{}_C{}_F{}_features.npy".format(identity,
                                                                          idx // 2 + 1,
                                                                          is_flipped))
                        this_feature_paths.append(this_feature_path)
                else:
                    crop_length = 1
                    this_feature_path = \
                        os.path.join(video_feature_folder, "{}_features.npy".format(identity))
                    this_feature_paths.append(this_feature_path)

                this_prediction_mean_path = os.path.join(video_feature_folder,
                                                    "{}_predictions.npy".format(identity))

                check_paths = this_feature_paths + [this_prediction_mean_path]

                if self.dataset_type == "training":
                    this_prediction_var_path = os.path.join(video_feature_folder,
                                                        "{}_variances.npy".format(identity))
                    check_paths += [this_prediction_var_path]

                existing_flag = True
                for path in check_paths:
                    if not os.path.exists(path):
                        existing_flag = False
                        break

                if existing_flag:
                    make_batch_index += 1
                    print("Forwarding ... Video {} {:04d} / {:04d} Pass".format(identity,
                                                                                make_batch_index,
                                                                                self.make_data.data_count))
                    continue

                entire_predictions = list()
                one_frame = cv2.imread(os.path.join(self.dataset.frames_folder, identity,
                                                    "images", "img_00001.jpg"))
                height, width, _ = one_frame.shape
                total_crop_height = (height - self.input_size[1])
                total_crop_width = (width - self.input_size[0])
                if self.dataset_type == "training":
                    crop_tops = np.linspace(0, total_crop_height, self.num_crops, dtype=np.int32)
                    crop_lefts = np.linspace(0, total_crop_width, self.num_crops, dtype=np.int32)
                    crop_product_space = list(itertools.product(crop_tops, crop_lefts))
                else:
                    crop_product_space = [(total_crop_height // 2, total_crop_width // 2)]

                for crop_index in range(crop_length):
                    batch_tf_data = []
                    for batch_temporal_index in range(len(first_frames[0])):
                        for batch_index in range(len(identities)):
                            tf_datum = "{} {} {} {} {}".format(
                                identities[batch_index].decode(),
                                first_frames[batch_index][batch_temporal_index],
                                crop_product_space[crop_index // 2][0],
                                crop_product_space[crop_index // 2][1],
                                int(crop_index % 2 == 1))

                            batch_tf_data.append(tf_datum)

                    data_session = tf.Session(graph=tf.Graph())
                    with data_session.as_default():
                        with data_session.graph.as_default():
                            if self.make_type == "RGB":
                                batch_data = tf.data.Dataset.from_tensor_slices(batch_tf_data)
                                batch_data = batch_data.prefetch(5 * self.step_size)
                                batch_data = \
                                    batch_data.map(lambda batch_datum:
                                                   tf.py_func(self.make_data.rgb_preprocessing_np,
                                                              [batch_datum], tf.float32),
                                                   num_parallel_calls=self.dataset.networks.num_workers)
                                batch_data = batch_data.batch(self.step_size)
                                batch_data = batch_data.prefetch(5)
                                batch_iterator = batch_data.make_one_shot_iterator()
                                batch_next_element = batch_iterator.get_next()

                            if self.make_type == "Flow":
                                batch_data = tf.data.Dataset.from_tensor_slices(batch_tf_data)
                                batch_data = batch_data.prefetch(5 * self.step_size)
                                batch_data = \
                                    batch_data.map(lambda batch_datum:
                                                   tf.py_func(self.make_data.flow_preprocessing_np,
                                                              [batch_datum], tf.float32),
                                                   num_parallel_calls=self.dataset.networks.num_workers)
                                batch_data = batch_data.batch(self.step_size)
                                batch_data = batch_data.prefetch(5)
                                batch_iterator = batch_data.make_one_shot_iterator()
                                batch_next_element = batch_iterator.get_next()

                            if self.make_type == "Fusion":
                                batch_data = tf.data.Dataset.from_tensor_slices(batch_tf_data)
                                batch_data = batch_data.prefetch(5 * self.step_size)
                                batch_data = \
                                    batch_data.map(lambda batch_datum:
                                                   tf.py_func(self.make_data.fusion_preprocessing_np,
                                                              [batch_datum], [tf.float32, tf.float32]),
                                                   num_parallel_calls=self.dataset.networks.num_workers)
                                batch_data = batch_data.batch(self.step_size)
                                batch_data = batch_data.prefetch(5)
                                batch_iterator = batch_data.make_one_shot_iterator()
                                batch_next_element = batch_iterator.get_next()

                    step_index = 0

                    if self.make_type in ["RGB", "Fusion"]:
                        if self.dformat == "NDHWC":
                            rgb_frames = np.zeros(dtype=np.float32,
                                                  shape=(self.temporal_width,
                                                         self.input_size[0],
                                                         self.input_size[1],
                                                         3))
                        else:
                            rgb_frames = np.zeros(dtype=np.float32,
                                                  shape=(3,
                                                         self.temporal_width,
                                                         self.input_size[0],
                                                         self.input_size[1]))

                    if self.make_type in ["Flow", "Fusion"]:
                        if self.dformat == "NDHWC":
                            flow_frames = np.zeros(dtype=np.float32,
                                                   shape=(self.temporal_width,
                                                          self.input_size[0],
                                                          self.input_size[1],
                                                          2))
                        else:
                            flow_frames = np.zeros(dtype=np.float32,
                                                   shape=(2,
                                                          self.temporal_width,
                                                          self.input_size[0],
                                                          self.input_size[1]))

                    volume_length = \
                        int(math.ceil(float(len(first_frames[0])) / float(self.step_size))) + \
                        (self.temporal_width // self.step_size - 1) * 2
                    features = np.zeros(dtype=np.float32,
                                        shape=(volume_length,
                                               1024 if self.make_type in ["RGB", "Flow"] else 2048))
                    predictions = np.zeros(dtype=np.float32, shape=(volume_length, self.dataset.number_of_classes - 1))

                    while True:
                        try:
                            if self.make_type == "RGB":
                                rgb_frame_vectors = \
                                    data_session.run(batch_next_element)
                            if self.make_type == "Flow":
                                flow_frame_vectors = \
                                    data_session.run(batch_next_element)
                            if self.make_type == "Fusion":
                                rgb_frame_vectors, flow_frame_vectors = \
                                    data_session.run(batch_next_element)
                        except tf.errors.OutOfRangeError:
                            break

                        if self.make_type in ["RGB", "Fusion"]:
                            rgb_frame_vector_length = rgb_frame_vectors.shape[0]
                            if rgb_frame_vector_length < self.step_size:
                                rgb_frame_vectors = \
                                    np.concatenate([rgb_frame_vectors,
                                                    np.zeros(dtype=np.float32,
                                                             shape=(self.step_size - rgb_frame_vector_length,
                                                                    self.input_size[0], self.input_size[1], 3))],
                                                   axis=0)

                            if self.dformat == "NDHWC":
                                rgb_frames[:-self.step_size] = rgb_frames[self.step_size:]
                                rgb_frames[-self.step_size:] = rgb_frame_vectors
                            else:
                                rgb_frame_vectors = np.transpose(rgb_frame_vectors, (3, 0, 1, 2))
                                rgb_frames[:, :-self.step_size] = rgb_frames[:, self.step_size:]
                                rgb_frames[:, -self.step_size:] = rgb_frame_vectors

                        if self.make_type in ["Flow", "Fusion"]:
                            flow_frame_vector_length = flow_frame_vectors.shape[0]
                            if flow_frame_vector_length < self.step_size:
                                flow_frame_vectors = \
                                    np.concatenate([flow_frame_vectors,
                                                    np.zeros(dtype=np.float32,
                                                             shape=(self.step_size - flow_frame_vector_length,
                                                                    self.input_size[0], self.input_size[1], 2))],
                                                   axis=0)

                            if self.dformat == "NDHWC":
                                flow_frames[:-self.step_size] = flow_frames[self.step_size:]
                                flow_frames[-self.step_size:] = flow_frame_vectors
                            else:
                                flow_frame_vectors = np.transpose(flow_frame_vectors, (3, 0, 1, 2))
                                flow_frames[:, :-self.step_size] = flow_frames[:, self.step_size:]
                                flow_frames[:, -self.step_size:] = flow_frame_vectors

                        if self.make_type == "RGB":
                            rgb_features, rgb_predictions = \
                                session.run([self.spotting_network_rgb.features,
                                             self.spotting_network_rgb.dense_predictions],
                                            feed_dict={
                                                self.spotting_network_rgb.frames:
                                                    np.expand_dims(rgb_frames, axis=0)
                                            })

                            this_features = np.squeeze(rgb_features, axis=0)
                            this_predictions = np.squeeze(rgb_predictions, axis=0)
                        elif self.make_type == "Flow":
                            flow_features, flow_predictions = \
                                session.run([self.spotting_network_flow.features,
                                             self.spotting_network_flow.dense_predictions],
                                            feed_dict={
                                                self.spotting_network_flow.frames:
                                                    np.expand_dims(flow_frames, axis=0)
                                            })

                            this_features = np.squeeze(flow_features, axis=0)
                            this_predictions = np.squeeze(flow_predictions, axis=0)
                        else:
                            rgb_features, rgb_predictions, \
                            flow_features, flow_predictions = \
                                session.run([self.spotting_network_rgb.features,
                                             self.spotting_network_rgb.dense_predictions,
                                             self.spotting_network_flow.features,
                                             self.spotting_network_flow.dense_predictions],
                                            feed_dict={
                                                self.spotting_network_rgb.frames:
                                                    np.expand_dims(rgb_frames, axis=0),
                                                self.spotting_network_flow.frames:
                                                    np.expand_dims(flow_frames, axis=0)})

                            this_features = \
                                np.concatenate([np.squeeze(rgb_features, axis=0),
                                                np.squeeze(flow_features, axis=0)],
                                               axis=-1)
                            this_predictions = (np.squeeze(rgb_predictions, axis=0) +
                                                np.squeeze(flow_predictions, axis=0)) / 2.0

                        if this_predictions.shape[-1] != self.dataset.number_of_classes - 1:
                            this_predictions = np.transpose(this_predictions, [1, 0])
                        start_index = step_index
                        end_index = step_index + (self.temporal_width // self.step_size)
                        features[start_index:end_index] += this_features
                        predictions[start_index:end_index] += this_predictions

                        step_index += 1
                        entire_length = volume_length + additional_loop - \
                                        ((self.temporal_width // self.step_size - 1)) * 2
                        print_string = \
                            "Forwarding|Video {}({:5d}/{:5d})|C({:2d}/{:2d})|T({:5d}/{:5d})".format(
                                identity,
                                make_batch_index + 1,
                                self.make_data.data_count,
                                crop_index + 1,
                                crop_length,
                                step_index,
                                entire_length)
                        progress_step = step_index + (crop_index) * entire_length
                        progress_length = entire_length * crop_length
                        print_string += \
                            " |{}{}|".format(
                                "=" * int(round(25.0 * float(progress_step) / float(progress_length))),
                                " " * (25 - int(round(25.0 * float(progress_step) / float(progress_length)))))
                        sys.stdout.write("\r" + print_string)
                        sys.stdout.flush()

                    for _ in range(additional_loop):
                        if self.make_type in ["RGB", "Fusion"]:
                            if self.dformat == "NDHWC":
                                rgb_frames[:-self.step_size] = rgb_frames[self.step_size:]
                                rgb_frames[-self.step_size:] = 0.0
                            else:
                                rgb_frames[:, :-self.step_size] = rgb_frames[:, self.step_size:]
                                rgb_frames[:, -self.step_size:] = 0.0

                        if self.make_type in ["Flow", "Fusion"]:
                            if self.dformat == "NDHWC":
                                flow_frames[:-self.step_size] = flow_frames[self.step_size:]
                                flow_frames[-self.step_size:] = 0.0
                            else:
                                flow_frames[:, :-self.step_size] = flow_frames[:, self.step_size:]
                                flow_frames[:, -self.step_size:] = 0.0

                        if self.make_type == "RGB":
                            rgb_features, rgb_predictions = \
                                session.run([self.spotting_network_rgb.features,
                                             self.spotting_network_rgb.dense_predictions],
                                            feed_dict={
                                                self.spotting_network_rgb.frames:
                                                    np.expand_dims(rgb_frames, axis=0)
                                            })

                            this_features = np.squeeze(rgb_features, axis=0)
                            this_predictions = np.squeeze(rgb_predictions, axis=0)
                        elif self.make_type == "Flow":
                            flow_features, flow_predictions = \
                                session.run([self.spotting_network_flow.features,
                                             self.spotting_network_flow.dense_predictions],
                                            feed_dict={
                                                self.spotting_network_flow.frames:
                                                    np.expand_dims(flow_frames, axis=0)
                                            })

                            this_features = np.squeeze(flow_features, axis=0)
                            this_predictions = np.squeeze(flow_predictions, axis=0)
                        else:
                            rgb_features, rgb_predictions, \
                            flow_features, flow_predictions = \
                                session.run([self.spotting_network_rgb.features,
                                             self.spotting_network_rgb.dense_predictions,
                                             self.spotting_network_flow.features,
                                             self.spotting_network_flow.dense_predictions],
                                            feed_dict={
                                                self.spotting_network_rgb.frames:
                                                    np.expand_dims(rgb_frames, axis=0),
                                                self.spotting_network_flow.frames:
                                                    np.expand_dims(flow_frames, axis=0)
                                            })

                            this_features = \
                                np.concatenate([np.squeeze(rgb_features, axis=0),
                                                np.squeeze(flow_features, axis=0)],
                                               axis=-1)
                            this_predictions = (np.squeeze(rgb_predictions, axis=0) +
                                                np.squeeze(flow_predictions, axis=0)) / 2.0

                        if this_predictions.shape[-1] != self.dataset.number_of_classes - 1:
                            this_predictions = np.transpose(this_predictions, [1, 0])
                        start_index = step_index
                        end_index = step_index + (self.temporal_width // self.step_size)
                        features[start_index:end_index] += this_features
                        predictions[start_index:end_index] += this_predictions

                        step_index += 1
                        entire_length = volume_length + additional_loop - \
                                        ((self.temporal_width // self.step_size - 1)) * 2
                        print_string = \
                            "Forwarding|Video {}({:5d}/{:5d})|C({:2d}/{:2d})|T({:5d}/{:5d})".format(
                                identity,
                                make_batch_index + 1,
                                self.make_data.data_count,
                                crop_index + 1,
                                crop_length,
                                step_index,
                                entire_length)
                        progress_step = step_index  + (crop_index) * entire_length
                        progress_length = entire_length * crop_length
                        print_string += \
                            " |{}{}|".format(
                                "=" * int(round(25.0 * float(progress_step) / float(progress_length))),
                                " " * (25 - int(round(25.0 * float(progress_step) / float(progress_length)))))
                        sys.stdout.write("\r" + print_string)
                        sys.stdout.flush()

                    del batch_tf_data, batch_data, batch_iterator
                    # if self.make_type in ["RGB", "Fusion"]:
                    #     del rgb_batch_next_element
                    # if self.make_type in ["Flow", "Fusion"]:
                    #     del flow_batch_next_element
                    del batch_next_element
                    data_session.close()
                    del data_session

                    features = np.divide(features, float(self.temporal_width // self.step_size))
                    if self.temporal_width != self.step_size:
                        features = \
                            features[self.temporal_width // self.step_size - 1:
                                     -(self.temporal_width // self.step_size - 1)]
                    predictions = np.divide(predictions, float(self.temporal_width // self.step_size))
                    if self.temporal_width != self.step_size:
                        predictions = \
                            predictions[self.temporal_width // self.step_size - 1:
                                        -(self.temporal_width // self.step_size - 1)]

                    np.save(this_feature_paths[crop_index], features)
                    entire_predictions.append(predictions)

                    del features, predictions
                    gc.collect()

                entire_predictions = np.array(entire_predictions)
                predictions_mean = np.mean(entire_predictions, axis=0)
                predictions_var = np.sqrt(np.var(entire_predictions, axis=0))
                np.save(this_prediction_mean_path, predictions_mean)

                if self.dataset_type == "training":
                    np.save(this_prediction_var_path, predictions_var)

                if self.dataset_type != "testing":
                    frame_length = len(glob.glob(os.path.join(self.dataset.frames_folder, identity, "images", "*")))
                    annotations = self.dataset.meta_dic["database"][identity]["annotations"]
                    # C*, f_s, f_e
                    ground_truths = list()
                    for annotation in annotations:
                        target = self.dataset.category_dic[annotation["label"].replace(" ", "_")]
                        segment = annotation["segment"]

                        start_index = max(1, int(math.floor(segment[0] * self.dataset.video_fps)))
                        end_index = min(int(math.ceil(segment[1] * self.dataset.video_fps)), frame_length)

                        if end_index - start_index + 1 >= 1:
                            ground_truths.append((target, start_index, end_index))

                    image_save_path = os.path.join(self.visualization_folder,
                                                   "{}.png".format(identity))
                    self.visualize(ground_truths, predictions_mean, predictions_var,
                                   frame_length, image_save_path)

                print("")
                make_batch_index += 1
                del entire_predictions, predictions_mean, predictions_var
                gc.collect()

    def make_kinetics_features(self):
        print("=" * 90)
        print("Networks Making")
        print("=" * 90)

        self.is_server = True
        self.num_workers = 20 if self.is_server else 8
        self.num_gpus = 2 if self.is_server else 1
        self.dataset_name = "thumos14"
        self.make_type = "RGB"

        self.num_crops = 5 if self.dataset_name == "thumos14" else 3
        self.batch_size = 1
        self.max_batch_size = 64
        self.temporal_width = 64
        self.validation_temporal_width = self.temporal_width
        self.step_size = 64

        self.dtype = tf.float32
        self.dformat = "NCDHW"
        self.data_type = "images"
        self.flow_type = "tvl1"
        self.dataset = self.Dataset(self)
        self.dataset_type = "training"
        self.make_data = self.dataset.getDataset("make", self.dataset_type)

        self.make_data.tf_dataset = self.make_data.tf_dataset.batch(1)
        self.make_data.tf_dataset = self.make_data.tf_dataset.prefetch(5)
        self.make_iterator = self.make_data.tf_dataset.make_initializable_iterator()
        self.make_next_element = self.make_iterator.get_next()

        if self.make_type in ["RGB", "Fusion"]:
            self.kinetics_load_rgb_ckpt_file_path = \
                os.path.join(self.dataset.root_path, "cnn", "I3D", "rgb", "model.ckpt")
            self.kinetics_load_rgb_npz_file_path = \
                os.path.join(self.dataset.root_path, "cnn", "I3D", "pretrained", "rgb_imagenet", "model.npz")

        if self.make_type in ["Flow", "Fusion"]:
            self.kinetics_load_flow_ckpt_file_path = \
                os.path.join(self.dataset.root_path, "cnn", "I3D", "flow", "model.ckpt")
            self.kinetics_load_flow_npz_file_path = \
                os.path.join(self.dataset.root_path, "cnn", "I3D", "pretrained", "flow_imagenet", "model.npz")

        if self.make_type == "Fusion":
            self.data_folder = \
                os.path.join(self.dataset.dataset_folder,
                             "Kinetics_Data_{}_{}".format(self.dataset_name,
                                                          "1027"))
        else:
            self.data_folder = \
                os.path.join(self.dataset.dataset_folder,
                             "Kinetics_Data_{}_{}".format(self.dataset_name,
                                                          "1027_{}".format(self.make_type)))

        self.feature_folder = os.path.join(self.data_folder, "features")
        self.visualization_folder = os.path.join(self.data_folder, "visualization")

        folders = [self.data_folder,
                   self.feature_folder,
                   self.visualization_folder]

        for folder in folders:
            if not os.path.exists(folder):
                try:
                    os.mkdir(folder)
                except OSError:
                    pass

        if self.num_gpus >= 2:
            if self.make_type == "Fusion":
                device_ids = [0, 1]
            else:
                device_ids = [0, 0]
        else:
            device_ids = [0, 0]

        batch_size = -1 if self.batch_size > self.max_batch_size else self.batch_size
        if self.make_type in ["RGB", "Fusion"]:
            self.model_name = "SpottingNetwork_RGB"
            self.spotting_network_rgb = self.SpottingNetwork(self, is_training=False,
                                                             data_type="images",
                                                             batch_size=batch_size,
                                                             device_id=device_ids[0],
                                                             num_classes=400)
            self.spotting_network_rgb.build_model()

        if self.make_type in ["Flow", "Fusion"]:
            self.model_name = "SpottingNetwork_Flow"
            self.spotting_network_flow = self.SpottingNetwork(self, is_training=False,
                                                              data_type="flows",
                                                              batch_size=batch_size,
                                                              device_id=device_ids[1],
                                                              num_classes=400)
            self.spotting_network_flow.build_model()

        os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(device_id) for device_id in range(self.num_gpus)])
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"

        self.rgb_logit_w = None
        self.rgb_logit_b = None
        self.rgb_load_parameters = dict()
        self.flow_logit_w = None
        self.flow_logit_b = None
        self.flow_load_parameters = dict()
        for param in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            if "I3D_RGB" in param.name:
                key_name = param.name[:-2].replace("SpottingNetwork_RGB/", "")
                self.rgb_load_parameters[key_name] = param

            if "I3D_Flow" in param.name:
                key_name = param.name[:-2].replace("SpottingNetwork_Flow/", "")
                self.flow_load_parameters[key_name] = param

            if "RGB" in param.name and "Logits" in param.name and "kernel" in param.name:
                self.rgb_logit_w = param

            if "RGB" in param.name and "Logits" in param.name and "bias" in param.name:
                self.rgb_logit_b = param

            if "Flow" in param.name and "Logits" in param.name and "kernel" in param.name:
                self.flow_logit_w = param

            if "Flow" in param.name and "Logits" in param.name and "bias" in param.name:
                self.flow_logit_b = param

        if self.make_type in ["RGB", "Fusion"]:
            rgb_cnn_loader = tf.train.Saver(var_list=self.rgb_load_parameters)

        if self.make_type in ["Flow", "Fusion"]:
            flow_cnn_loader = tf.train.Saver(var_list=self.flow_load_parameters)

        self.labels = dict()
        with open(os.path.join(self.dataset.meta_folder, "kinetics-400_classes.txt"), "r") as fp:
            while True:
                line = fp.readline()
                splits = line[:-1].split()
                if len(splits) < 2:
                    break
                category = splits[0]
                class_number = splits[1]
                self.labels[class_number] = category

        cmap = plt.get_cmap("terrain")
        cNorm = matplotlib.colors.Normalize(vmin=0, vmax=400)
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)

        self.color_maps = scalarMap

        with tf.Session() as session:
            print("Loading Pre-trained Models ...")
            if self.make_type in ["RGB", "Fusion"]:
                rgb_cnn_loader.restore(session, self.kinetics_load_rgb_ckpt_file_path)
                rgb_logit_params = np.load(self.kinetics_load_rgb_npz_file_path)
                session.run(tf.assign(self.rgb_logit_w,
                                      rgb_logit_params["Logits/Conv3d_0c_1x1/conv_3d/w"]))
                session.run(tf.assign(self.rgb_logit_b,
                                      np.reshape(rgb_logit_params["Logits/Conv3d_0c_1x1/conv_3d/b"],
                                                 (1, 1, 1, 1, 400) if self.dformat == "NDHWC"
                                                 else (1, 400, 1, 1, 1))))
            if self.make_type in ["Flow", "Fusion"]:
                flow_cnn_loader.restore(session, self.kinetics_load_flow_ckpt_file_path)
                flow_logit_params = np.load(self.kinetics_load_flow_npz_file_path)
                session.run(tf.assign(self.flow_logit_w,
                                      flow_logit_params["Logits/Conv3d_0c_1x1/conv_3d/w"]))
                session.run(tf.assign(self.flow_logit_b,
                                      np.reshape(flow_logit_params["Logits/Conv3d_0c_1x1/conv_3d/b"],
                                                 (1, 1, 1, 1, 400) if self.dformat == "NDHWC"
                                                 else (1, 400, 1, 1, 1))))
            print("Pre-trained Models are Loaded!")

            make_batch_index = 0
            session.run(self.make_iterator.initializer)

            additional_loop = self.temporal_width // self.step_size - 1

            while True:
                try:
                    identities, first_frames, \
                    first_targets, crop_settings, \
                    segments = session.run(self.make_next_element)
                except tf.errors.OutOfRangeError:
                    break

                identity = str(identities[0].decode())

                video_feature_folder = os.path.join(self.feature_folder, identity)
                if not os.path.exists(video_feature_folder):
                    try:
                        os.mkdir(video_feature_folder)
                    except OSError:
                        pass

                this_feature_paths = list()
                if self.dataset_type == "training":
                    crop_length = self.num_crops * self.num_crops * 2
                    for idx in range(crop_length):
                        is_flipped = int(idx % 2 == 1)
                        this_feature_path = \
                            os.path.join(video_feature_folder,
                                         "{}_C{}_F{}_features.npy".format(identity,
                                                                          idx // 2 + 1,
                                                                          is_flipped))
                        this_feature_paths.append(this_feature_path)
                else:
                    crop_length = 1
                    this_feature_path = \
                        os.path.join(video_feature_folder, "{}_features.npy".format(identity))
                    this_feature_paths.append(this_feature_path)
                this_prediction_mean_path = os.path.join(video_feature_folder,
                                                    "{}_predictions.npy".format(identity))
                this_prediction_var_path = os.path.join(video_feature_folder,
                                                    "{}_variances.npy".format(identity))

                check_paths = this_feature_paths + \
                              [this_prediction_mean_path, this_prediction_var_path]
                existing_flag = True
                for path in check_paths:
                    if not os.path.exists(path):
                        existing_flag = False
                        break

                if existing_flag:
                    make_batch_index += 1
                    print("Forwarding ... Video {} {:04d} / {:04d} Pass".format(identity,
                                                                                make_batch_index,
                                                                                self.make_data.data_count))
                    continue

                entire_predictions = list()
                # if self.dataset_name == "thumos14":
                one_frame = cv2.imread(os.path.join(self.dataset.frames_folder, identity,
                                                    "images", "img_00001.jpg"))
                height, width, _ = one_frame.shape
                total_crop_height = (height - self.input_size[1])
                total_crop_width = (width - self.input_size[0])
                if self.dataset_type == "training":
                    crop_tops = np.linspace(0, total_crop_height, self.num_crops, dtype=np.int32)
                    crop_lefts = np.linspace(0, total_crop_width, self.num_crops, dtype=np.int32)
                    crop_product_space = list(itertools.product(crop_tops, crop_lefts))
                else:
                    crop_product_space = [(total_crop_height // 2, total_crop_width // 2)]
                for crop_index in range(crop_length):
                    batch_tf_data = []
                    for batch_temporal_index in range(len(first_frames[0])):
                        for batch_index in range(len(identities)):
                            # if self.dataset_name == "thumos14":
                            tf_datum = "{} {} {} {} {}".format(
                                identities[batch_index].decode(),
                                first_frames[batch_index][batch_temporal_index],
                                crop_product_space[crop_index // 2][0],
                                crop_product_space[crop_index // 2][1],
                                int(crop_index % 2 == 1))
                            # else:
                            #     tf_datum = "{} {} {} {} {}".format(
                            #         identities[batch_index].decode(),
                            #         first_frames[batch_index][batch_temporal_index],
                            #         remaining_crop_product_space[crop_index // 2][0],
                            #         remaining_crop_product_space[crop_index // 2][1],
                            #         int(crop_index % 2 == 1))

                            batch_tf_data.append(tf_datum)

                    data_session = tf.Session(graph=tf.Graph())
                    with data_session.as_default():
                        with data_session.graph.as_default():
                            if self.make_type == "RGB":
                                batch_data = tf.data.Dataset.from_tensor_slices(batch_tf_data)
                                batch_data = batch_data.prefetch(5 * self.step_size)
                                batch_data = \
                                    batch_data.map(lambda batch_datum:
                                                   tf.py_func(self.make_data.rgb_preprocessing_np,
                                                              [batch_datum], tf.float32),
                                                   num_parallel_calls=self.dataset.networks.num_workers)
                                # batch_data = batch_data.map(self.make_data.rgb_preprocessing,
                                #                             num_parallel_calls=self.num_workers)
                                batch_data = batch_data.batch(self.step_size)
                                batch_data = batch_data.prefetch(5)
                                batch_iterator = batch_data.make_one_shot_iterator()
                                batch_next_element = batch_iterator.get_next()

                            if self.make_type == "Flow":
                                batch_data = tf.data.Dataset.from_tensor_slices(batch_tf_data)
                                batch_data = batch_data.prefetch(5 * self.step_size)
                                batch_data = \
                                    batch_data.map(lambda batch_datum:
                                                   tf.py_func(self.make_data.flow_preprocessing_np,
                                                              [batch_datum], tf.float32),
                                                   num_parallel_calls=self.dataset.networks.num_workers)
                                # batch_data = batch_data.map(self.make_data.flow_preprocessing,
                                #                             num_parallel_calls=self.num_workers)
                                batch_data = batch_data.batch(self.step_size)
                                batch_data = batch_data.prefetch(5)
                                batch_iterator = batch_data.make_one_shot_iterator()
                                batch_next_element = batch_iterator.get_next()

                            if self.make_type == "Fusion":
                                batch_data = tf.data.Dataset.from_tensor_slices(batch_tf_data)
                                batch_data = batch_data.prefetch(5 * self.step_size)
                                batch_data = \
                                    batch_data.map(lambda batch_datum:
                                                   tf.py_func(self.make_data.fusion_preprocessing_np,
                                                              [batch_datum], [tf.float32, tf.float32]),
                                                   num_parallel_calls=self.dataset.networks.num_workers)
                                # batch_data = batch_data.map(self.make_data.flow_preprocessing,
                                #                             num_parallel_calls=self.num_workers)
                                batch_data = batch_data.batch(self.step_size)
                                batch_data = batch_data.prefetch(5)
                                batch_iterator = batch_data.make_one_shot_iterator()
                                batch_next_element = batch_iterator.get_next()

                    step_index = 0

                    if self.make_type in ["RGB", "Fusion"]:
                        if self.dformat == "NDHWC":
                            rgb_frames = np.zeros(dtype=np.float32,
                                                  shape=(self.temporal_width,
                                                         self.input_size[0],
                                                         self.input_size[1],
                                                         3))
                        else:
                            rgb_frames = np.zeros(dtype=np.float32,
                                                  shape=(3,
                                                         self.temporal_width,
                                                         self.input_size[0],
                                                         self.input_size[1]))

                    if self.make_type in ["Flow", "Fusion"]:
                        if self.dformat == "NDHWC":
                            flow_frames = np.zeros(dtype=np.float32,
                                                   shape=(self.temporal_width,
                                                          self.input_size[0],
                                                          self.input_size[1],
                                                          2))
                        else:
                            flow_frames = np.zeros(dtype=np.float32,
                                                   shape=(2,
                                                          self.temporal_width,
                                                          self.input_size[0],
                                                          self.input_size[1]))

                    volume_length = \
                        int(math.ceil(float(len(first_frames[0])) / float(self.step_size))) + \
                        (self.temporal_width // self.step_size - 1) * 2
                    features = np.zeros(dtype=np.float32,
                                        shape=(volume_length,
                                               1024 if self.make_type in ["RGB", "Flow"] else 2048))
                    predictions = np.zeros(dtype=np.float32,
                                           shape=(volume_length, 400))

                    while True:
                        try:
                            if self.make_type == "RGB":
                                rgb_frame_vectors = \
                                    data_session.run(batch_next_element)
                            if self.make_type == "Flow":
                                flow_frame_vectors = \
                                    data_session.run(batch_next_element)
                            if self.make_type == "Fusion":
                                rgb_frame_vectors, flow_frame_vectors = \
                                    data_session.run(batch_next_element)
                        except tf.errors.OutOfRangeError:
                            break

                        if self.make_type in ["RGB", "Fusion"]:
                            rgb_frame_vector_length = rgb_frame_vectors.shape[0]
                            if rgb_frame_vector_length < self.step_size:
                                rgb_frame_vectors = \
                                    np.concatenate([rgb_frame_vectors,
                                                    np.zeros(dtype=np.float32,
                                                             shape=(self.step_size - rgb_frame_vector_length,
                                                                    self.input_size[0], self.input_size[1], 3))],
                                                   axis=0)

                            if self.dformat == "NDHWC":
                                rgb_frames[:-self.step_size] = rgb_frames[self.step_size:]
                                rgb_frames[-self.step_size:] = rgb_frame_vectors
                            else:
                                rgb_frame_vectors = np.transpose(rgb_frame_vectors, (3, 0, 1, 2))
                                rgb_frames[:, :-self.step_size] = rgb_frames[:, self.step_size:]
                                rgb_frames[:, -self.step_size:] = rgb_frame_vectors

                        if self.make_type in ["Flow", "Fusion"]:
                            flow_frame_vector_length = flow_frame_vectors.shape[0]
                            if flow_frame_vector_length < self.step_size:
                                flow_frame_vectors = \
                                    np.concatenate([flow_frame_vectors,
                                                    np.zeros(dtype=np.float32,
                                                             shape=(self.step_size - flow_frame_vector_length,
                                                                    self.input_size[0], self.input_size[1], 2))],
                                                   axis=0)

                            if self.dformat == "NDHWC":
                                flow_frames[:-self.step_size] = flow_frames[self.step_size:]
                                flow_frames[-self.step_size:] = flow_frame_vectors
                            else:
                                flow_frame_vectors = np.transpose(flow_frame_vectors, (3, 0, 1, 2))
                                flow_frames[:, :-self.step_size] = flow_frames[:, self.step_size:]
                                flow_frames[:, -self.step_size:] = flow_frame_vectors

                        if self.make_type == "RGB":
                            rgb_features, rgb_predictions = \
                                session.run([self.spotting_network_rgb.features,
                                             self.spotting_network_rgb.dense_predictions],
                                            feed_dict={
                                                self.spotting_network_rgb.frames:
                                                    np.expand_dims(rgb_frames, axis=0)
                                            })

                            this_features = np.squeeze(rgb_features, axis=0)
                            this_predictions = np.squeeze(rgb_predictions, axis=0)
                        elif self.make_type == "Flow":
                            flow_features, flow_predictions = \
                                session.run([self.spotting_network_flow.features,
                                             self.spotting_network_flow.dense_predictions],
                                            feed_dict={
                                                self.spotting_network_flow.frames:
                                                    np.expand_dims(flow_frames, axis=0)
                                            })

                            this_features = np.squeeze(flow_features, axis=0)
                            this_predictions = np.squeeze(flow_predictions, axis=0)
                        else:
                            rgb_features, rgb_predictions, \
                            flow_features, flow_predictions = \
                                session.run([self.spotting_network_rgb.features,
                                             self.spotting_network_rgb.dense_predictions,
                                             self.spotting_network_flow.features,
                                             self.spotting_network_flow.dense_predictions],
                                            feed_dict={
                                                self.spotting_network_rgb.frames:
                                                    np.expand_dims(rgb_frames, axis=0),
                                                self.spotting_network_flow.frames:
                                                    np.expand_dims(flow_frames, axis=0)
                                            })

                            this_features = \
                                np.concatenate([np.squeeze(rgb_features, axis=0),
                                                np.squeeze(flow_features, axis=0)],
                                               axis=-1)
                            this_predictions = (np.squeeze(rgb_predictions, axis=0) +
                                                np.squeeze(flow_predictions, axis=0)) / 2.0

                        if this_predictions.shape[-1] != 400:
                            this_predictions = np.transpose(this_predictions, [1, 0])
                        start_index = step_index
                        end_index = step_index + (self.temporal_width // self.step_size)
                        features[start_index:end_index] += this_features
                        predictions[start_index:end_index] += this_predictions

                        step_index += 1
                        entire_length = volume_length + additional_loop - \
                                        ((self.temporal_width // self.step_size - 1)) * 2
                        print_string = \
                            "Forwarding|Video {}({:5d}/{:5d})|C({:2d}/{:2d})|T({:5d}/{:5d})".format(
                                identity,
                                make_batch_index + 1,
                                self.make_data.data_count,
                                crop_index + 1,
                                crop_length,
                                step_index,
                                entire_length)
                        progress_step = step_index + (crop_index) * entire_length
                        progress_length = entire_length * crop_length
                        print_string += \
                            " |{}{}|".format(
                                "=" * int(round(25.0 * float(progress_step) / float(progress_length))),
                                " " * (25 - int(round(25.0 * float(progress_step) / float(progress_length)))))
                        sys.stdout.write("\r" + print_string)
                        sys.stdout.flush()

                    for _ in range(additional_loop):
                        if self.make_type in ["RGB", "Fusion"]:
                            if self.dformat == "NDHWC":
                                rgb_frames[:-self.step_size] = rgb_frames[self.step_size:]
                                rgb_frames[-self.step_size:] = 0.0
                            else:
                                rgb_frames[:, :-self.step_size] = rgb_frames[:, self.step_size:]
                                rgb_frames[:, -self.step_size:] = 0.0

                        if self.make_type in ["Flow", "Fusion"]:
                            if self.dformat == "NDHWC":
                                flow_frames[:-self.step_size] = flow_frames[self.step_size:]
                                flow_frames[-self.step_size:] = 0.0
                            else:
                                flow_frames[:, :-self.step_size] = flow_frames[:, self.step_size:]
                                flow_frames[:, -self.step_size:] = 0.0

                        if self.make_type == "RGB":
                            rgb_features, rgb_predictions = \
                                session.run([self.spotting_network_rgb.features,
                                             self.spotting_network_rgb.dense_predictions],
                                            feed_dict={
                                                self.spotting_network_rgb.frames:
                                                    np.expand_dims(rgb_frames, axis=0)
                                            })

                            this_features = np.squeeze(rgb_features, axis=0)
                            this_predictions = np.squeeze(rgb_predictions, axis=0)
                        elif self.make_type == "Flow":
                            flow_features, flow_predictions = \
                                session.run([self.spotting_network_flow.features,
                                             self.spotting_network_flow.dense_predictions],
                                            feed_dict={
                                                self.spotting_network_flow.frames:
                                                    np.expand_dims(flow_frames, axis=0)
                                            })

                            this_features = np.squeeze(flow_features, axis=0)
                            this_predictions = np.squeeze(flow_predictions, axis=0)
                        else:
                            rgb_features, rgb_predictions, \
                            flow_features, flow_predictions = \
                                session.run([self.spotting_network_rgb.features,
                                             self.spotting_network_rgb.dense_predictions,
                                             self.spotting_network_flow.features,
                                             self.spotting_network_flow.dense_predictions],
                                            feed_dict={
                                                self.spotting_network_rgb.frames:
                                                    np.expand_dims(rgb_frames, axis=0),
                                                self.spotting_network_flow.frames:
                                                    np.expand_dims(flow_frames, axis=0)
                                            })

                            this_features = \
                                np.concatenate([np.squeeze(rgb_features, axis=0),
                                                np.squeeze(flow_features, axis=0)],
                                               axis=-1)
                            this_predictions = (np.squeeze(rgb_predictions, axis=0) +
                                                np.squeeze(flow_predictions, axis=0)) / 2.0

                        if this_predictions.shape[-1] != 400:
                            this_predictions = np.transpose(this_predictions, [1, 0])
                        start_index = step_index
                        end_index = step_index + (self.temporal_width // self.step_size)
                        features[start_index:end_index] += this_features
                        predictions[start_index:end_index] += this_predictions

                        step_index += 1
                        entire_length = volume_length + additional_loop - \
                                        ((self.temporal_width // self.step_size - 1)) * 2
                        print_string = \
                            "Forwarding|Video {}({:5d}/{:5d})|C({:2d}/{:2d})|T({:5d}/{:5d})".format(
                                identity,
                                make_batch_index + 1,
                                self.make_data.data_count,
                                crop_index + 1,
                                crop_length,
                                step_index,
                                entire_length)
                        progress_step = step_index  + (crop_index) * entire_length
                        progress_length = entire_length * crop_length
                        print_string += \
                            " |{}{}|".format(
                                "=" * int(round(25.0 * float(progress_step) / float(progress_length))),
                                " " * (25 - int(round(25.0 * float(progress_step) / float(progress_length)))))
                        sys.stdout.write("\r" + print_string)
                        sys.stdout.flush()

                    del batch_tf_data, batch_data, batch_iterator
                    # if self.make_type in ["RGB", "Fusion"]:
                    #     del rgb_batch_next_element
                    # if self.make_type in ["Flow", "Fusion"]:
                    #     del flow_batch_next_element
                    del batch_next_element
                    data_session.close()
                    del data_session

                    features = np.divide(features, float(self.temporal_width // self.step_size))
                    features = \
                        features[self.temporal_width // self.step_size - 1:
                                 -(self.temporal_width // self.step_size - 1)]
                    predictions = np.divide(predictions, float(self.temporal_width // self.step_size))
                    predictions = \
                        predictions[self.temporal_width // self.step_size - 1:
                                    -(self.temporal_width // self.step_size - 1)]

                    np.save(this_feature_paths[crop_index], features)
                    entire_predictions.append(predictions)

                    del features, predictions
                    gc.collect()

                entire_predictions = np.array(entire_predictions)
                predictions_mean = np.mean(entire_predictions, axis=0)
                predictions_var = np.sqrt(np.var(entire_predictions, axis=0))

                np.save(this_prediction_mean_path, predictions_mean)
                np.save(this_prediction_var_path, predictions_var)

                # if self.dataset_type != "testing":
                #     frame_length = len(glob.glob(os.path.join(self.dataset.frames_folder, identity, "images", "*")))
                #     annotations = self.dataset.meta_dic["database"][identity]["annotations"]
                #     # C*, f_s, f_e
                #     ground_truths = list()
                #     for annotation in annotations:
                #         target = self.dataset.category_dic[annotation["label"].replace(" ", "_")]
                #         segment = annotation["segment"]
                #
                #         start_index = max(1, int(math.floor(segment[0] * self.dataset.video_fps)))
                #         end_index = min(int(math.ceil(segment[1] * self.dataset.video_fps)), frame_length)
                #
                #         if end_index - start_index + 1 >= 1:
                #             ground_truths.append((target, start_index, end_index))
                #
                #     image_save_path = os.path.join(self.visualization_folder,
                #                                    "{}.png".format(identity))
                #     self.visualize(ground_truths, predictions_mean, predictions_var,
                #                    frame_length, image_save_path)

                print("")
                make_batch_index += 1
                del entire_predictions, predictions_mean, predictions_var
                gc.collect()

    def make_single_crop(self):
        print("=" * 90)
        print("Networks Making")
        print("=" * 90)

        self.is_server = True
        self.num_workers = 20 if self.is_server else 8
        self.num_gpus = 2 if self.is_server else 1
        self.dataset_name = "activitynet"
        self.make_type = "Fusion"

        self.batch_size = 1
        self.temporal_width = 64
        self.validation_temporal_width = self.temporal_width
        self.step_size = 8

        self.dtype = tf.float32
        self.dformat = "NCDHW"
        self.data_type = "images"
        self.flow_type = "tvl1"
        self.dataset = self.Dataset(self)
        self.dataset_type = "validation"
        self.make_data = self.dataset.getDataset("make", self.dataset_type)

        self.make_data.tf_dataset = self.make_data.tf_dataset.batch(1)
        self.make_data.tf_dataset = self.make_data.tf_dataset.prefetch(5)
        self.make_iterator = self.make_data.tf_dataset.make_initializable_iterator()
        self.make_next_element = self.make_iterator.get_next()

        if self.make_type in ["RGB", "Fusion"]:
            self.load_rgb_ckpt_file_path = \
                os.path.join(self.dataset.root_path, "networks", "weights", "restore",
                             "SpottingNetwork_RGB_activitynet_0423", "weights.ckpt-50")

        if self.make_type in ["Flow", "Fusion"]:
            self.load_flow_ckpt_file_path = \
                os.path.join(self.dataset.root_path, "networks", "weights", "restore",
                             "SpottingNetwork_Flow_activitynet_0427", "weights.ckpt-80")

        if self.make_type == "Fusion":
            self.data_folder = \
                os.path.join(self.dataset.dataset_folder,
                             "Finetuned_Data_{}_{}".format(self.dataset_name,
                                                           "0427"))
        else:
            self.data_folder = \
                os.path.join(self.dataset.dataset_folder,
                             "Finetuned_Data_{}_{}".format(self.dataset_name,
                                                           "0427_{}".format(self.make_type)))

        self.feature_folder = os.path.join(self.data_folder, "features")
        self.visualization_folder = os.path.join(self.data_folder, "visualization")

        folders = [self.data_folder,
                   self.feature_folder,
                   self.visualization_folder]

        for folder in folders:
            if not os.path.exists(folder):
                try:
                    os.mkdir(folder)
                except OSError:
                    pass

        if self.num_gpus >= 2:
            if self.make_type == "Fusion":
                device_ids = [0, 1]
            else:
                device_ids = [0, 0]
        else:
            device_ids = [0, 0]

        if self.make_type in ["RGB", "Fusion"]:
            self.model_name = "SpottingNetwork_RGB"
            self.spotting_network_rgb = self.SpottingNetwork(self, is_training=False, data_type="images",
                                                             batch_size=self.batch_size,
                                                             device_id=device_ids[0])
            self.spotting_network_rgb.build_model()

        if self.make_type in ["Flow", "Fusion"]:
            self.model_name = "SpottingNetwork_Flow"
            self.spotting_network_flow = self.SpottingNetwork(self, is_training=False, data_type="flows",
                                                              batch_size=self.batch_size,
                                                              device_id=device_ids[1])
            self.spotting_network_flow.build_model()

        os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(device_id) for device_id in range(self.num_gpus)])
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"

        if self.make_type in ["RGB", "Fusion"]:
            spotting_rgb_loader = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                            scope="SpottingNetwork_RGB"))

        if self.make_type in ["Flow", "Fusion"]:
            spotting_flow_loader = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                             scope="SpottingNetwork_Flow"))

        cmap = plt.get_cmap("terrain")
        cNorm = matplotlib.colors.Normalize(vmin=0, vmax=self.dataset.number_of_classes)
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)

        self.color_maps = scalarMap

        with tf.Session() as session:
            print("Loading Pre-trained Models ...")
            if self.make_type in ["RGB", "Fusion"]:
                spotting_rgb_loader.restore(session, self.spotting_load_rgb_ckpt_file_path)
            if self.make_type in ["Flow", "Fusion"]:
                spotting_flow_loader.restore(session, self.spotting_load_flow_ckpt_file_path)
            print("Pre-trained Models are Loaded!")

            make_batch_index = 0
            session.run(self.make_iterator.initializer)

            additional_loop = self.temporal_width // self.step_size - 1

            while True:
                try:
                    identities, first_frames, \
                    first_targets, crop_settings, \
                    segments = session.run(self.make_next_element)
                except tf.errors.OutOfRangeError:
                    break

                identity = str(identities[0].decode())

                video_feature_folder = os.path.join(self.feature_folder, identity)
                if not os.path.exists(video_feature_folder):
                    try:
                        os.mkdir(video_feature_folder)
                    except OSError:
                        pass

                this_feature_path = os.path.join(video_feature_folder,
                                                 "{}_features.npy".format(identity))
                this_prediction_path = os.path.join(video_feature_folder,
                                                    "{}_predictions.npy".format(identity))

                check_paths = [this_feature_path, this_prediction_path]
                existing_flag = True
                for path in check_paths:
                    if not os.path.exists(path):
                        existing_flag = False
                        break

                if existing_flag:
                    make_batch_index += 1
                    print("Forwarding ... Video {} {:04d} / {:04d} Pass".format(identity,
                                                                                make_batch_index,
                                                                                self.make_data.data_count))
                    continue

                batch_tf_data = []
                for batch_temporal_index in range(len(first_frames[0])):
                    for batch_index in range(len(identities)):
                        tf_datum = "{} {:05d} {} {} {}".format(identities[batch_index].decode(),
                                                               first_frames[batch_index][
                                                                   batch_temporal_index],
                                                               first_targets[batch_index][
                                                                   batch_temporal_index],
                                                               crop_settings[batch_index][0],
                                                               crop_settings[batch_index][1])

                        batch_tf_data.append(tf_datum)

                data_session = tf.Session(graph=tf.Graph())
                with data_session.as_default():
                    with data_session.graph.as_default():
                        if self.make_type in ["RGB", "Fusion"]:
                            batch_data = tf.data.Dataset.from_tensor_slices(batch_tf_data)
                            batch_data = batch_data.prefetch(self.step_size * 2)
                            batch_data = batch_data.map(self.make_data.rgb_preprocessing,
                                                        num_parallel_calls=self.num_workers)
                            batch_data = batch_data.batch(self.step_size)
                            batch_data = batch_data.prefetch(5)
                            batch_iterator = batch_data.make_one_shot_iterator()
                            rgb_batch_next_element = batch_iterator.get_next()

                        if self.make_type in ["Flow", "Fusion"]:
                            batch_data = tf.data.Dataset.from_tensor_slices(batch_tf_data)
                            batch_data = batch_data.prefetch(self.step_size * 2)
                            batch_data = batch_data.map(self.make_data.flow_preprocessing,
                                                        num_parallel_calls=self.num_workers)
                            batch_data = batch_data.batch(self.step_size)
                            batch_data = batch_data.prefetch(5)
                            batch_iterator = batch_data.make_one_shot_iterator()
                            flow_batch_next_element = batch_iterator.get_next()

                step_index = 0

                if self.make_type in ["RGB", "Fusion"]:
                    if self.dformat == "NDHWC":
                        rgb_frames = np.zeros(dtype=np.float32,
                                              shape=(self.temporal_width,
                                                     self.input_size[0],
                                                     self.input_size[1],
                                                     3))
                    else:
                        rgb_frames = np.zeros(dtype=np.float32,
                                              shape=(3,
                                                     self.temporal_width,
                                                     self.input_size[0],
                                                     self.input_size[1]))

                if self.make_type in ["Flow", "Fusion"]:
                    if self.dformat == "NDHWC":
                        flow_frames = np.zeros(dtype=np.float32,
                                               shape=(self.temporal_width,
                                                      self.input_size[0],
                                                      self.input_size[1],
                                                      2))
                    else:
                        flow_frames = np.zeros(dtype=np.float32,
                                               shape=(2,
                                                      self.temporal_width,
                                                      self.input_size[0],
                                                      self.input_size[1]))

                volume_length = \
                    int(math.ceil(float(len(first_frames[0])) / float(self.step_size))) + \
                    (self.temporal_width // self.step_size - 1) * 2
                features = np.zeros(dtype=np.float32,
                                    shape=(volume_length,
                                           1024 if self.make_type in ["RGB", "Flow"] else 2048))
                predictions = np.zeros(dtype=np.float32,
                                       shape=(volume_length, self.dataset.number_of_classes))

                while True:
                    try:
                        if self.make_type in ["RGB", "Fusion"]:
                            rgb_frame_vectors, target_vectors = data_session.run(rgb_batch_next_element)
                        if self.make_type in ["Flow", "Fusion"]:
                            flow_frame_vectors, target_vectors = data_session.run(flow_batch_next_element)
                    except tf.errors.OutOfRangeError:
                        break

                    if self.make_type in ["RGB", "Fusion"]:
                        rgb_frame_vector_length = len(rgb_frame_vectors)
                        if rgb_frame_vector_length < self.step_size:
                            rgb_frame_vectors = \
                                np.concatenate([rgb_frame_vectors,
                                                np.zeros(dtype=np.float32,
                                                         shape=(self.step_size - rgb_frame_vector_length,
                                                                self.input_size[0], self.input_size[1], 3))],
                                               axis=0)

                        if self.dformat == "NDHWC":
                            rgb_frames[:-self.step_size] = rgb_frames[self.step_size:]
                            rgb_frames[-self.step_size:] = rgb_frame_vectors
                        else:
                            rgb_frame_vectors = np.transpose(rgb_frame_vectors, (3, 0, 1, 2))
                            rgb_frames[:, :-self.step_size] = rgb_frames[:, self.step_size:]
                            rgb_frames[:, -self.step_size:] = rgb_frame_vectors

                    if self.make_type in ["Flow", "Fusion"]:
                        flow_frame_vector_length = len(flow_frame_vectors)
                        if flow_frame_vector_length < self.step_size:
                            flow_frame_vectors = \
                                np.concatenate([flow_frame_vectors,
                                                np.zeros(dtype=np.float32,
                                                         shape=(self.step_size - flow_frame_vector_length,
                                                                self.input_size[0], self.input_size[1], 2))],
                                               axis=0)

                        if self.dformat == "NDHWC":
                            flow_frames[:-self.step_size] = flow_frames[self.step_size:]
                            flow_frames[-self.step_size:] = flow_frame_vectors
                        else:
                            flow_frame_vectors = np.transpose(flow_frame_vectors, (3, 0, 1, 2))
                            flow_frames[:, :-self.step_size] = flow_frames[:, self.step_size:]
                            flow_frames[:, -self.step_size:] = flow_frame_vectors

                    if self.make_type == "RGB":
                        rgb_features, rgb_predictions = \
                            session.run([self.spotting_network_rgb.features,
                                         self.spotting_network_rgb.dense_predictions],
                                        feed_dict={
                                            self.spotting_network_rgb.frames: np.expand_dims(rgb_frames, 0)
                                        })

                        this_features = rgb_features
                        this_predictions = rgb_predictions
                    elif self.make_type == "Flow":
                        flow_features, flow_predictions = \
                            session.run([self.spotting_network_flow.features,
                                         self.spotting_network_flow.dense_predictions],
                                        feed_dict={
                                            self.spotting_network_flow.frames: np.expand_dims(flow_frames, 0)
                                        })

                        this_features = flow_features
                        this_predictions = flow_predictions
                    else:
                        rgb_features, rgb_predictions, \
                        flow_features, flow_predictions = \
                            session.run([self.spotting_network_rgb.features,
                                         self.spotting_network_rgb.dense_predictions,
                                         self.spotting_network_flow.features,
                                         self.spotting_network_flow.dense_predictions],
                                        feed_dict={
                                            self.spotting_network_rgb.frames: np.expand_dims(rgb_frames, 0),
                                            self.spotting_network_flow.frames: np.expand_dims(flow_frames, 0)
                                        })

                        this_features = np.concatenate([rgb_features, flow_features], axis=-1)
                        this_predictions = (rgb_predictions + flow_predictions) / 2.0

                    start_index = step_index
                    end_index = step_index + (self.temporal_width // self.step_size - 1)
                    features[start_index:end_index] += this_features
                    predictions[start_index:end_index] += this_predictions

                    step_index += 1
                    entire_length = volume_length + additional_loop
                    print_string = \
                        "Forwarding|Video {}({:5d}/{:5d})|{:5d}/{:5d}".format(identity,
                                                                              make_batch_index + 1,
                                                                              self.make_data.data_count,
                                                                              step_index,
                                                                              entire_length)
                    print_string += \
                        " |{}{}|".format("=" * int(round(25.0 * float(step_index) / float(entire_length))),
                                         " " * (25 - int(round(25.0 * float(step_index) / float(entire_length)))))
                    sys.stdout.write("\r" + print_string)
                    sys.stdout.flush()

                for _ in range(additional_loop):
                    if self.make_type in ["RGB", "Fusion"]:
                        if self.dformat == "NDHWC":
                            rgb_frames[:-self.step_size] = rgb_frames[self.step_size:]
                            rgb_frames[-self.step_size:] = 0.0
                        else:
                            rgb_frames[:, :-self.step_size] = rgb_frames[:, self.step_size:]
                            rgb_frames[:, -self.step_size:] = 0.0

                    if self.make_type in ["Flow", "Fusion"]:
                        if self.dformat == "NDHWC":
                            flow_frames[:-self.step_size] = flow_frames[self.step_size:]
                            flow_frames[-self.step_size:] = 0.0
                        else:
                            flow_frames[:, :-self.step_size] = flow_frames[:, self.step_size:]
                            flow_frames[:, -self.step_size:] = 0.0

                    if self.make_type == "RGB":
                        rgb_features, rgb_predictions = \
                            session.run([self.spotting_network_rgb.features,
                                         self.spotting_network_rgb.dense_predictions],
                                        feed_dict={
                                            self.spotting_network_rgb.frames: np.expand_dims(rgb_frames, 0)
                                        })

                        this_features = rgb_features
                        this_predictions = rgb_predictions
                    elif self.make_type == "Flow":
                        flow_features, flow_predicitons = \
                            session.run([self.spotting_network_flow.features,
                                         self.spotting_network_flow.dense_predictions],
                                        feed_dict={
                                            self.spotting_network_flow.frames: np.expand_dims(flow_frames, 0)
                                        })

                        this_features = flow_features
                        this_predictions = flow_predictions
                    else:
                        rgb_features, rgb_predictions, \
                        flow_features, flow_predictions = \
                            session.run([self.spotting_network_rgb.features,
                                         self.spotting_network_rgb.dense_predictions,
                                         self.spotting_network_flow.features,
                                         self.spotting_network_flow.dense_predictions],
                                        feed_dict={
                                            self.spotting_network_rgb.frames: np.expand_dims(rgb_frames, 0),
                                            self.spotting_network_flow.frames: np.expand_dims(flow_frames, 0)
                                        })

                        this_features = np.concatenate([rgb_features, flow_features], axis=-1)
                        this_predictions = (rgb_predictions + flow_predictions) / 2.0

                    start_index = step_index
                    end_index = step_index + (self.temporal_width // self.step_size - 1)
                    features[start_index:end_index] += this_features
                    predictions[start_index:end_index] += this_predictions

                    step_index += 1
                    entire_length = volume_length + additional_loop
                    print_string = \
                        "Forwarding|Video {}({:5d}/{:5d})|{:5d}/{:5d}".format(identity,
                                                                              make_batch_index + 1,
                                                                              self.make_data.data_count,
                                                                              step_index,
                                                                              entire_length)
                    print_string += \
                        " |{}{}|".format("=" * int(round(25.0 * float(step_index) / float(entire_length))),
                                         " " * (25 - int(round(25.0 * float(step_index) / float(entire_length)))))
                    sys.stdout.write("\r" + print_string)
                    sys.stdout.flush()

                del batch_tf_data, batch_data, batch_iterator
                if self.make_type in ["RGB", "Fusion"]:
                    del rgb_batch_next_element
                if self.make_type in ["Flow", "Fusion"]:
                    del flow_batch_next_element
                data_session.close()
                del data_session

                features = np.divide(features, float(self.temporal_width // self.step_size))
                predictions = np.divide(predictions, float(self.temporal_width // self.step_size))

                np.save(this_feature_path, features)
                np.save(this_prediction_path, predictions)

                if self.dataset_type != "testing":
                    frame_length = len(glob.glob(os.path.join(self.dataset.frames_folder, identity, "images", "*")))
                    annotations = self.dataset.meta_dic["database"][identity]["annotations"]
                    # C*, f_s, f_e
                    ground_truths = list()
                    for annotation in annotations:
                        target = self.dataset.category_dic[annotation["label"].replace(" ", "_")]
                        segment = annotation["segment"]

                        start_index = max(1, int(math.floor(segment[0] * self.dataset.video_fps)))
                        end_index = min(int(math.ceil(segment[1] * self.dataset.video_fps)), frame_length)

                        if end_index - start_index + 1 >= 1:
                            ground_truths.append((target, start_index, end_index))

                    image_save_path = os.path.join(self.visualization_folder,
                                                   "{}.png".format(identity))
                    self.visualize(ground_truths, predictions, frame_length, image_save_path)

                print("")
                make_batch_index += 1
                del features, predictions
                gc.collect()

    def make_centered(self):
        print("=" * 90)
        print("Networks Making")
        print("=" * 90)

        self.is_server = True
        self.num_workers = 20 if self.is_server else 8
        self.num_gpus = 2 if self.is_server else 1
        self.dataset_name = "thumos14"
        self.temporal_width = 64
        self.make_type = "Fusion"

        self.validation_batch_size = 1
        self.validation_size = -1
        self.validation_term = 1
        self.validation_temporal_width = self.temporal_width
        self.validation_temporal_sample_size = -1
        self.validation_display_term = 1
        self.validation_running_batch = 1
        self.validation_step_size = 2

        self.dtype = tf.float32
        self.dformat = "NCDHW"
        self.data_type = "images"
        self.flow_type = "tvl1"
        self.dataset = self.Dataset(self)
        self.dataset_type = "all"
        self.validation_data = self.dataset.getDataset("make")

        self.validation_data.tf_dataset = self.validation_data.tf_dataset.batch(self.validation_batch_size)
        self.validation_data.tf_dataset = self.validation_data.tf_dataset.prefetch(5)
        self.validation_iterator = self.validation_data.tf_dataset.make_initializable_iterator()
        self.validation_next_element = self.validation_iterator.get_next()

        if self.make_type in ["RGB", "Fusion"]:
            self.spotting_load_rgb_ckpt_file_path = \
                os.path.join(self.dataset.root_path, "networks", "weights", "restore",
                             "SpottingNetwork_RGB_{}_0916".format(self.dataset_name), "weights.ckpt-200")

        if self.make_type in ["Flow", "Fusion"]:
            self.spotting_load_flow_ckpt_file_path = \
                os.path.join(self.dataset.root_path, "networks", "weights", "restore",
                             "SpottingNetwork_Flow_{}_0917".format(self.dataset_name), "weights.ckpt-300")

        if self.make_type == "Fusion":
            self.spotting_forwarding_data_folder = \
                os.path.join(self.dataset.dataset_folder,
                             "SpottingNetwork_ForwardingData_{}_{}".format(self.dataset_name,
                                                                           "0917"))
        else:
            self.spotting_forwarding_data_folder = \
                os.path.join(self.dataset.dataset_folder,
                             "SpottingNetwork_ForwardingData_{}_{}".format(self.dataset_name,
                                                                           "0917_{}".format(self.make_type)))

        self.spotting_feature_folder = os.path.join(self.spotting_forwarding_data_folder, "features")
        self.spotting_result_folder = os.path.join(self.spotting_forwarding_data_folder, "results")
        self.validation_image_folder = self.spotting_result_folder

        folders = [self.spotting_forwarding_data_folder,
                   self.spotting_feature_folder,
                   self.spotting_result_folder]

        for folder in folders:
            if not os.path.exists(folder):
                try:
                    os.mkdir(folder)
                except OSError:
                    pass

        if self.num_gpus >= 2:
            running_batch_sizes = [self.validation_running_batch, self.validation_running_batch]
            if self.make_type == "Fusion":
                device_ids = [0, 1]
            else:
                device_ids = [None, None]

        if self.make_type in ["RGB", "Fusion"]:
            self.model_name = "SpottingNetwork_RGB"
            self.spotting_network_rgb = self.SpottingNetwork(self, is_training=False, data_type="images",
                                                             batch_size=running_batch_sizes[0],
                                                             device_id=device_ids[0])
            self.spotting_network_rgb.build_model()

        if self.make_type in ["Flow", "Fusion"]:
            self.model_name = "SpottingNetwork_Flow"
            self.spotting_network_flow = self.SpottingNetwork(self, is_training=False, data_type="flows",
                                                              batch_size=running_batch_sizes[1],
                                                              device_id=device_ids[1])
            self.spotting_network_flow.build_model()

        os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(device_id) for device_id in range(self.num_gpus)])
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"

        if self.make_type in ["RGB", "Fusion"]:
            spotting_rgb_loader = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                            scope="SpottingNetwork_RGB"))

        if self.make_type in ["Flow", "Fusion"]:
            spotting_flow_loader = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                             scope="SpottingNetwork_Flow"))

        cmap = plt.get_cmap("terrain")
        cNorm = matplotlib.colors.Normalize(vmin=0, vmax=self.dataset.number_of_classes)
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)

        self.color_maps = scalarMap

        with tf.Session() as session:
            print("Loading Pre-trained Models ...")
            if self.make_type in ["RGB", "Fusion"]:
                spotting_rgb_loader.restore(session, self.spotting_load_rgb_ckpt_file_path)
            if self.make_type in ["Flow", "Fusion"]:
                spotting_flow_loader.restore(session, self.spotting_load_flow_ckpt_file_path)
            print("Pre-trained Models are Loaded!")

            validation_batch_index = 0
            session.run(self.validation_iterator.initializer)

            while True:
                try:
                    identities, first_frames, \
                    first_targets, crop_settings, \
                    segments = \
                        session.run(self.validation_next_element)
                except tf.errors.OutOfRangeError:
                    break

                identity = str(identities[0])

                video_feature_folder = os.path.join(self.spotting_feature_folder, identity)
                if not os.path.exists(video_feature_folder):
                    try:
                        os.mkdir(video_feature_folder)
                    except OSError:
                        pass

                this_prediction_path = os.path.join(video_feature_folder,
                                                    "{}_predictions.npy".format(identity))
                this_feature_path = os.path.join(video_feature_folder,
                                                 "{}_features.npy".format(identity))

                check_paths = [this_prediction_path, this_feature_path]
                existing_flag = True
                for path in check_paths:
                    if not os.path.exists(path):
                        existing_flag = False
                        break

                if existing_flag:
                    validation_batch_index += 1
                    print("Forwarding ... Video {} {:04d} / {:04d} Pass".format(identity,
                                                                                validation_batch_index,
                                                                                self.validation_data.data_count))
                    continue

                batch_tf_data = []
                for batch_temporal_index in range(len(first_frames[0])):
                    for batch_index in range(len(identities)):
                        tf_datum = "{} {:05d} {} {} {}".format(identities[batch_index],
                                                               first_frames[batch_index][
                                                                   batch_temporal_index],
                                                               first_targets[batch_index][
                                                                   batch_temporal_index],
                                                               crop_settings[batch_index][0],
                                                               crop_settings[batch_index][1])

                        batch_tf_data.append(tf_datum)

                data_session = tf.Session(graph=tf.Graph())
                with data_session.as_default():
                    with data_session.graph.as_default():
                        if self.make_type in ["RGB", "Fusion"]:
                            batch_data = tf.data.Dataset.from_tensor_slices(batch_tf_data)
                            batch_data = batch_data.prefetch(self.validation_step_size * 2)
                            batch_data = batch_data.map(self.validation_data.rgb_preprocessing,
                                                        num_parallel_calls=self.num_workers)
                            batch_data = batch_data.batch(self.validation_step_size)
                            batch_data = batch_data.prefetch(2)
                            batch_iterator = batch_data.make_one_shot_iterator()
                            rgb_batch_next_element = batch_iterator.get_next()

                        if self.make_type in ["Flow", "Fusion"]:
                            batch_data = tf.data.Dataset.from_tensor_slices(batch_tf_data)
                            batch_data = batch_data.prefetch(self.validation_step_size * 2)
                            batch_data = batch_data.map(self.validation_data.flow_preprocessing,
                                                        num_parallel_calls=self.num_workers)
                            batch_data = batch_data.batch(self.validation_step_size)
                            batch_data = batch_data.prefetch(2)
                            batch_iterator = batch_data.make_one_shot_iterator()
                            flow_batch_next_element = batch_iterator.get_next()

                step_index = 0

                if self.make_type in ["RGB", "Fusion"]:
                    rgb_frames = list()
                if self.make_type in ["Flow", "Fusion"]:
                    flow_frames = list()

                volume_length = int(math.ceil(float(len(first_frames[0])) / float(self.validation_step_size)))
                features = np.zeros(dtype=np.float32,
                                    shape=(volume_length,
                                           1024 if self.make_type in ["RGB", "Flow"] else 2048))
                predictions = np.zeros(dtype=np.float32,
                                       shape=(volume_length, self.dataset.number_of_classes))

                initial_out_flag, kept_out_flag, final_out_flag = False, False, False
                while True:
                    try:
                        if self.make_type in ["RGB", "Fusion"]:
                            rgb_frame_vectors, target_vectors = data_session.run(rgb_batch_next_element)
                        if self.make_type in ["Flow", "Fusion"]:
                            flow_frame_vectors, target_vectors = data_session.run(flow_batch_next_element)
                    except tf.errors.OutOfRangeError:
                        break

                    if step_index == 0:
                        # for 8 step size, 36 8 20
                        # initial loop ((64 - 8) / 2 - 8) / 8 = 3
                        # 3 * 8 = 24 --> 20, 4(kept), so 3 loops needed.
                        # initial_loop --> 3
                        initial_loop = \
                            int(math.ceil(float((self.validation_temporal_width - self.validation_step_size) / 2 -
                                                self.validation_step_size) /
                                          float(self.validation_step_size)))

                        # make 36 zero frames for 8 step size
                        for _ in range((self.validation_temporal_width - self.validation_step_size) // 2 +
                                       self.validation_step_size):
                            if self.make_type in ["RGB", "Fusion"]:
                                rgb_frames.append(np.zeros_like(rgb_frame_vectors[0]))
                            if self.make_type in ["Flow", "Fusion"]:
                                flow_frames.append(np.zeros_like(flow_frame_vectors[0]))

                        # already received frames are added
                        if self.make_type in ["RGB", "Fusion"]:
                            rgb_frames.extend(rgb_frame_vectors)
                        if self.make_type in ["Flow", "Fusion"]:
                            flow_frames.extend(flow_frame_vectors)

                        # 2 loops for 8 step size --> 16 frames
                        for _ in range(initial_loop - 1):
                            try:
                                if self.make_type in ["RGB", "Fusion"]:
                                    rgb_frame_vectors, target_vectors = data_session.run(rgb_batch_next_element)
                                if self.make_type in ["Flow", "Fusion"]:
                                    flow_frame_vectors, target_vectors = data_session.run(flow_batch_next_element)
                            except tf.errors.OutOfRangeError:
                                initial_out_flag = True
                                break
                            if self.make_type in ["RGB", "Fusion"]:
                                rgb_frames.extend(rgb_frame_vectors)
                            if self.make_type in ["Flow", "Fusion"]:
                                flow_frames.extend(flow_frame_vectors)

                        if not initial_out_flag:
                            # 8 frames --> 4, 4(kept)
                            try:
                                if self.make_type in ["RGB", "Fusion"]:
                                    rgb_frame_vectors, target_vectors = data_session.run(rgb_batch_next_element)
                                    rgb_frames.extend(rgb_frame_vectors[:self.validation_step_size // 2])
                                    kept_rgb_frame_vectors = rgb_frame_vectors[self.validation_step_size // 2:]
                                if self.make_type in ["Flow", "Fusion"]:
                                    flow_frame_vectors, target_vectors = data_session.run(flow_batch_next_element)
                                    flow_frames.extend(flow_frame_vectors[:self.validation_step_size // 2])
                                    kept_flow_frame_vectors = flow_frame_vectors[self.validation_step_size // 2:]
                            except tf.errors.OutOfRangeError:
                                kept_out_flag = True

                        if not (initial_out_flag + kept_out_flag):
                            try:
                                if self.make_type in ["RGB", "Fusion"]:
                                    rgb_frame_vectors, target_vectors = data_session.run(rgb_batch_next_element)
                                if self.make_type in ["Flow", "Fusion"]:
                                    flow_frame_vectors, target_vectors = data_session.run(flow_batch_next_element)
                            except tf.errors.OutOfRangeError:
                                final_out_flag = True

                        if initial_out_flag + kept_out_flag + final_out_flag:
                            if initial_out_flag:
                                if self.make_type in ["RGB", "Fusion"]:
                                    for _ in range(self.validation_temporal_width -
                                                   len(rgb_frames) - self.validation_step_size // 2):
                                        rgb_frames.append(np.zeros_like(rgb_frames[0]))

                                    kept_rgb_frame_vectors = \
                                        np.repeat(np.expand_dims(np.zeros_like(rgb_frames[0]),
                                                                 axis=0),
                                                  self.validation_step_size // 2,
                                                  axis=0)

                                    rgb_frame_vectors = \
                                        np.repeat(np.expand_dims(np.zeros_like(rgb_frames[0]),
                                                                 axis=0),
                                                  self.validation_step_size,
                                                  axis=0)

                                if self.make_type in ["Flow", "Fusion"]:
                                    for _ in range(self.validation_temporal_width -
                                                   len(flow_frames) - self.validation_step_size // 2):
                                        flow_frames.append(np.zeros_like(flow_frames[0]))

                                    kept_flow_frame_vectors = \
                                        np.repeat(np.expand_dims(np.zeros_like(flow_frames[0]),
                                                                 axis=0),
                                                  self.validation_step_size // 2,
                                                  axis=0)

                                    flow_frame_vectors = \
                                        np.repeat(np.expand_dims(np.zeros_like(flow_frames[0]),
                                                                 axis=0),
                                                  self.validation_step_size,
                                                  axis=0)

                            if not initial_out_flag and kept_out_flag:
                                if self.make_type in ["RGB", "Fusion"]:
                                    for _ in range(self.validation_step_size // 2):
                                        rgb_frames.append(np.zeros_like(rgb_frames[0]))

                                    kept_rgb_frame_vectors = \
                                        np.repeat(np.expand_dims(np.zeros_like(rgb_frames[0]),
                                                                 axis=0),
                                                  self.validation_step_size // 2,
                                                  axis=0)

                                    rgb_frame_vectors = \
                                        np.repeat(np.expand_dims(np.zeros_like(rgb_frames[0]),
                                                                 axis=0),
                                                  self.validation_step_size,
                                                  axis=0)

                                if self.make_type in ["Flow", "Fusion"]:
                                    for _ in range(self.validation_step_size // 2):
                                        flow_frames.append(np.zeros_like(flow_frames[0]))

                                    kept_flow_frame_vectors = \
                                        np.repeat(np.expand_dims(np.zeros_like(flow_frames[0]),
                                                                 axis=0),
                                                  self.validation_step_size // 2,
                                                  axis=0)

                                    flow_frame_vectors = \
                                        np.repeat(np.expand_dims(np.zeros_like(flow_frames[0]),
                                                                 axis=0),
                                                  self.validation_step_size,
                                                  axis=0)

                            if not (initial_out_flag + kept_out_flag) and final_out_flag:
                                if self.make_type in ["RGB", "Fusion"]:
                                    for _ in range(self.validation_temporal_width -
                                                   len(rgb_frames) - self.validation_step_size // 2):
                                        rgb_frames.append(np.zeros_like(rgb_frames[0]))

                                    if len(kept_rgb_frame_vectors) < self.validation_step_size // 2:
                                        kept_rgb_frame_vectors = \
                                            np.concatenate([kept_rgb_frame_vectors,
                                                            np.repeat(np.expand_dims(
                                                                np.zeros_like(kept_rgb_frame_vectors[0]),
                                                                axis=0),
                                                                self.validation_step_size // 2 -
                                                                len(kept_rgb_frame_vectors),
                                                                axis=0)],
                                                           axis=0)

                                    rgb_frame_vectors = \
                                        np.repeat(np.expand_dims(np.zeros_like(rgb_frames[0]),
                                                                 axis=0),
                                                  self.validation_step_size,
                                                  axis=0)

                                if self.make_type in ["Flow", "Fusion"]:
                                    for _ in range(self.validation_temporal_width -
                                                   len(flow_frames) - self.validation_step_size // 2):
                                        flow_frames.append(np.zeros_like(flow_frames[0]))

                                    if len(kept_flow_frame_vectors) < self.validation_step_size // 2:
                                        kept_flow_frame_vectors = \
                                            np.concatenate([kept_flow_frame_vectors,
                                                            np.repeat(np.expand_dims(
                                                                np.zeros_like(kept_flow_frame_vectors[0]),
                                                                axis=0),
                                                                self.validation_step_size // 2 -
                                                                len(kept_flow_frame_vectors),
                                                                axis=0)],
                                                           axis=0)

                                    flow_frame_vectors = \
                                        np.repeat(np.expand_dims(np.zeros_like(flow_frames[0]),
                                                                 axis=0),
                                                  self.validation_step_size,
                                                  axis=0)

                    if self.make_type in ["RGB", "Fusion"]:
                        del rgb_frames[:self.validation_step_size]
                        rgb_frames.extend(kept_rgb_frame_vectors)
                        rgb_frames.extend(rgb_frame_vectors[:self.validation_step_size // 2])
                        kept_rgb_frame_vectors = rgb_frame_vectors[self.validation_step_size // 2:]

                    if self.make_type in ["Flow", "Fusion"]:
                        del flow_frames[:self.validation_step_size]
                        flow_frames.extend(kept_flow_frame_vectors)
                        flow_frames.extend(flow_frame_vectors[:self.validation_step_size // 2])
                        kept_flow_frame_vectors = flow_frame_vectors[self.validation_step_size // 2:]

                    if self.make_type in ["RGB", "Fusion"]:
                        if len(rgb_frames) < self.validation_temporal_width:
                            for _ in range(self.validation_temporal_width - len(rgb_frames)):
                                rgb_frames.append(np.zeros_like(rgb_frames[0]))

                    if self.make_type in ["Flow", "Fusion"]:
                        if len(flow_frames) < self.validation_temporal_width:
                            for _ in range(self.validation_temporal_width - len(flow_frames)):
                                flow_frames.append(np.zeros_like(flow_frames[0]))

                    if self.dataset.networks.dformat == "NCDHW":
                        if self.make_type in ["RGB", "Fusion"]:
                            this_rgb_frames = np.transpose(np.array(rgb_frames), [3, 0, 1, 2])
                        if self.make_type in ["Flow", "Fusion"]:
                            this_flow_frames = np.transpose(np.array(flow_frames), [3, 0, 1, 2])
                    else:
                        if self.make_type in ["RGB", "Fusion"]:
                            this_rgb_frames = np.array(rgb_frames)
                        if self.make_type in ["Flow", "Fusion"]:
                            this_flow_frames = np.array(flow_frames)

                    if self.make_type == "RGB":
                        rgb_predictions, rgb_features = \
                            session.run([self.spotting_network_rgb.predictions,
                                         self.spotting_network_rgb.features],
                                        feed_dict={
                                            self.spotting_network_rgb.frames: [this_rgb_frames]
                                        })

                        this_predictions = rgb_predictions
                        this_features = rgb_features
                    elif self.make_type == "Flow":
                        flow_predictions, flow_features = \
                            session.run([self.spotting_network_flow.predictions,
                                         self.spotting_network_flow.features],
                                        feed_dict={
                                            self.spotting_network_flow.frames: [this_flow_frames]
                                        })

                        this_predictions = flow_predictions
                        this_features = flow_features
                    else:
                        rgb_predictions, rgb_features, \
                        flow_predictions, flow_features = \
                            session.run([self.spotting_network_rgb.predictions,
                                         self.spotting_network_rgb.features,
                                         self.spotting_network_flow.predictions,
                                         self.spotting_network_flow.features],
                                        feed_dict={
                                            self.spotting_network_rgb.frames: [this_rgb_frames],
                                            self.spotting_network_flow.frames: [this_flow_frames]
                                        })

                        this_predictions = (rgb_predictions + flow_predictions) / 2.0
                        this_features = np.concatenate([rgb_features, flow_features], axis=-1)

                    predictions[step_index] = this_predictions
                    features[step_index] = this_features

                    step_index += 1
                    print_string = \
                        "Forwarding|Video {}({:5d}/{:5d})|{:5d}/{:5d}".format(identity,
                                                                              validation_batch_index + 1,
                                                                              self.validation_data.data_count,
                                                                              step_index,
                                                                              volume_length)
                    print_string += \
                        " |{}{}|".format("=" * int(round(25.0 * float(step_index) / float(volume_length))),
                                         " " * (25 - int(round(25.0 * float(step_index) / float(volume_length)))))
                    sys.stdout.write("\r" + print_string)
                    sys.stdout.flush()

                additinal_loop = \
                    int(math.ceil(float((self.validation_temporal_width - self.validation_step_size) / 2) /
                                  float(self.validation_step_size)))

                for _ in range(additinal_loop):
                    if self.make_type in ["RGB", "Fusion"]:
                        rgb_frame_vectors = \
                            np.repeat(np.expand_dims(np.zeros_like(rgb_frames[0]),
                                                     axis=0),
                                      self.validation_step_size,
                                      axis=0)

                        if len(kept_rgb_frame_vectors) < self.validation_step_size // 2:
                            if len(kept_rgb_frame_vectors) >= 1:
                                kept_rgb_frame_vectors = \
                                    np.concatenate([kept_rgb_frame_vectors,
                                                    np.repeat(np.expand_dims(
                                                        np.zeros_like(kept_rgb_frame_vectors[0]),
                                                        axis=0),
                                                        self.validation_step_size // 2 -
                                                        len(kept_rgb_frame_vectors),
                                                        axis=0)],
                                                   axis=0)
                            else:
                                kept_rgb_frame_vectors = \
                                    np.repeat(np.expand_dims(np.zeros_like(rgb_frames[0]),
                                                             axis=0),
                                              self.validation_step_size // 2,
                                              axis=0)

                        del rgb_frames[:self.validation_step_size]
                        rgb_frames.extend(kept_rgb_frame_vectors)
                        rgb_frames.extend(rgb_frame_vectors[:self.validation_step_size // 2])
                        kept_rgb_frame_vectors = rgb_frame_vectors[self.validation_step_size // 2:]

                    if self.make_type in ["Flow", "Fusion"]:
                        flow_frame_vectors = \
                            np.repeat(np.expand_dims(np.zeros_like(flow_frames[0]),
                                                     axis=0),
                                      self.validation_step_size,
                                      axis=0)

                        if len(kept_flow_frame_vectors) < self.validation_step_size // 2:
                            if len(kept_flow_frame_vectors) >= 1:
                                kept_flow_frame_vectors = \
                                    np.concatenate([kept_flow_frame_vectors,
                                                    np.repeat(np.expand_dims(
                                                        np.zeros_like(kept_flow_frame_vectors[0]),
                                                        axis=0),
                                                        self.validation_step_size // 2 -
                                                        len(kept_flow_frame_vectors),
                                                        axis=0)],
                                                   axis=0)
                            else:
                                kept_flow_frame_vectors = \
                                    np.repeat(np.expand_dims(np.zeros_like(flow_frames[0]),
                                                             axis=0),
                                              self.validation_step_size // 2,
                                              axis=0)

                        del flow_frames[:self.validation_step_size]
                        flow_frames.extend(kept_flow_frame_vectors)
                        flow_frames.extend(flow_frame_vectors[:self.validation_step_size // 2])
                        kept_flow_frame_vectors = flow_frame_vectors[self.validation_step_size // 2:]

                    if self.dataset.networks.dformat == "NCDHW":
                        if self.make_type in ["RGB", "Fusion"]:
                            this_rgb_frames = np.transpose(np.array(rgb_frames), [3, 0, 1, 2])
                        if self.make_type in ["Flow", "Fusion"]:
                            this_flow_frames = np.transpose(np.array(flow_frames), [3, 0, 1, 2])
                    else:
                        if self.make_type in ["RGB", "Fusion"]:
                            this_rgb_frames = np.array(rgb_frames)
                        if self.make_type in ["Flow", "Fusion"]:
                            this_flow_frames = np.array(flow_frames)

                    if self.make_type == "RGB":
                        rgb_predictions, rgb_features = \
                            session.run([self.spotting_network_rgb.predictions,
                                         self.spotting_network_rgb.features],
                                        feed_dict={
                                            self.spotting_network_rgb.frames: [this_rgb_frames]
                                        })

                        this_predictions = rgb_predictions
                        this_features = rgb_features
                    elif self.make_type == "Flow":
                        flow_predictions, flow_features = \
                            session.run([self.spotting_network_flow.predictions,
                                         self.spotting_network_flow.features],
                                        feed_dict={
                                            self.spotting_network_flow.frames: [this_flow_frames]
                                        })

                        this_predictions = flow_predictions
                        this_features = flow_features
                    else:
                        rgb_predictions, rgb_features, \
                        flow_predictions, flow_features = \
                            session.run([self.spotting_network_rgb.predictions,
                                         self.spotting_network_rgb.features,
                                         self.spotting_network_flow.predictions,
                                         self.spotting_network_flow.features],
                                        feed_dict={
                                            self.spotting_network_rgb.frames: [this_rgb_frames],
                                            self.spotting_network_flow.frames: [this_flow_frames]
                                        })

                        this_predictions = (rgb_predictions + flow_predictions) / 2.0
                        this_features = np.concatenate([rgb_features, flow_features], axis=-1)

                    predictions[step_index] = this_predictions
                    features[step_index] = this_features

                    step_index += 1
                    print_string = \
                        "Forwarding|Video {}({:5d}/{:5d})|{:5d}/{:5d}".format(identity,
                                                                              validation_batch_index + 1,
                                                                              self.validation_data.data_count,
                                                                              step_index,
                                                                              volume_length)
                    print_string += \
                        " |{}{}|".format("=" * int(round(25.0 * float(step_index) / float(volume_length))),
                                         " " * (25 - int(round(25.0 * float(step_index) / float(volume_length)))))
                    sys.stdout.write("\r" + print_string)
                    sys.stdout.flush()

                del batch_tf_data, batch_data, batch_iterator
                if self.make_type in ["RGB", "Fusion"]:
                    del rgb_batch_next_element
                if self.make_type in ["Flow", "Fusion"]:
                    del flow_batch_next_element
                data_session.close()
                del data_session

                np.save(this_prediction_path, predictions)
                np.save(this_feature_path, features)

                self.each_visualize(identities[0])

                print("")
                validation_batch_index += 1
                del predictions, features
                gc.collect()

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

    def tsne(self):
        self.is_server = False
        self.dataset_name = "thumos14"
        self.data_type = "images"
        self.flow_type = "tvl1"
        self.temporal_width = 64
        self.feature_interval = 2

        self.dataset = self.Dataset(self)

        self.data_folder = os.path.join(self.dataset.dataset_folder,
                                        "SpottingNetwork_Data_{}_0925".format(self.dataset_name))
        self.feature_folder = os.path.join(self.data_folder, "features")
        self.tsne_folder = os.path.join(self.data_folder, "tsne")
        if not os.path.exists(self.tsne_folder):
            try:
                os.mkdir(self.tsne_folder)
            except OSError:
                pass

        mapping = \
            lambda x: int(round((float(x * self.feature_interval) + float((x + 1) * self.feature_interval)) / 2.0))

        entire_features = list()
        feature_labels = list()
        identities = self.dataset.meta_dic["database"].keys()
        for idx, identity in enumerate(identities):
            if self.dataset.meta_dic["database"][identity]["subset"] == "validation":
                features = np.load(os.path.join(self.feature_folder, identity,
                                                "{}_features.npy".format(identity)))
                predictions = np.load(os.path.join(self.feature_folder, identity,
                                                "{}_predictions.npy".format(identity)))

                frame_length = len(glob.glob(os.path.join(self.dataset.frames_folder, identity, "images", "*")))
                annotations = self.dataset.meta_dic["database"][identity]["annotations"]
                # C*, f_s, f_e
                ground_truths = list()
                for annotation in annotations:
                    target = self.dataset.category_dic[annotation["label"].replace(" ", "_")]
                    segment = annotation["segment"]

                    start_index = max(1, int(math.floor(segment[0] * self.dataset.video_fps)))
                    end_index = min(int(math.ceil(segment[1] * self.dataset.video_fps)), frame_length)

                    if end_index - start_index + 1 >= 1:
                        ground_truths.append((target, start_index, end_index))

                foreground_indices = list()
                background_indices = list()
                boundary_indices = list()
                for feature_idx in range(len(features)):
                    frame_idx = mapping(feature_idx)
                    matched_gt = None
                    is_boundary = False
                    for ground_truth in ground_truths:
                        target, start_idx, end_idx = ground_truth
                        if (frame_idx >= start_idx and frame_idx <= end_idx) or \
                            (frame_idx + 1 >= start_idx and frame_idx + 1 <= end_idx):
                            matched_gt = ground_truth
                            if (frame_idx == start_idx) or (frame_idx == end_idx) or \
                                    (frame_idx + 1 == start_idx) or (frame_idx + 1 == end_idx):
                                is_boundary = True

                    if matched_gt is None:
                        background_indices.append((feature_idx, np.argmax(predictions[feature_idx])))
                    elif is_boundary:
                        boundary_indices.append((feature_idx, np.argmax(predictions[feature_idx])))
                    else:
                        foreground_indices.append((feature_idx, matched_gt[0] - 1, np.argmax(predictions[feature_idx])))

                sampled_foreground_indices = \
                    random.sample(foreground_indices, min(10, len(foreground_indices)))
                sampled_background_indices = \
                    random.sample(background_indices, min(5, len(background_indices)))
                sampled_boundary_indices = \
                    random.sample(boundary_indices, min(5, len(boundary_indices)))

                for feature_idx in sampled_foreground_indices:
                    entire_features.append(features[feature_idx[0]])
                    # feature_labels.append(feature_idx[1] + 2)
                    feature_labels.append((feature_idx[1] + 2, feature_idx[2]))

                for feature_idx in sampled_background_indices:
                    entire_features.append(features[feature_idx[0]])
                    # feature_labels.append(0)
                    feature_labels.append((0, feature_idx[1]))

                for feature_idx in sampled_boundary_indices:
                    entire_features.append(features[feature_idx[0]])
                    feature_labels.append((1, feature_idx[1]))

            print("Generating regions ... {} {:5d}/{:5d} {:6.2f}%".format(identity, idx + 1, len(identities),
                                                                          float(idx + 1) /
                                                                          float(len(identities)) * 100.0))

        feature_labels = np.array(feature_labels)

        tsne = TSNE(n_components=2)
        transformed = tsne.fit_transform(np.array(entire_features))
        xs = transformed[:, 0]
        ys = transformed[:, 1]

        fig, axs = plt.subplots(1, 1, figsize=(30, 30))

        boundaries = (feature_labels[:, 0] == 1)
        non_boundaries = (feature_labels[:, 0] != 1)
        axs.scatter(xs[boundaries], ys[boundaries], marker="+", c=feature_labels[boundaries, 1])
        axs.scatter(xs[non_boundaries], ys[non_boundaries], c=feature_labels[non_boundaries, 1])

        for class_idx in range(self.dataset.number_of_classes - 1):
            class_indices = (feature_labels[:, 1] == class_idx + 1)
            class_xs = xs[class_indices]
            class_ys = ys[class_indices]
            self.confidence_ellipse(x=class_xs, y=class_ys, n_std=1.0, ax=axs, edgecolor="black")

        save_path = os.path.join(self.tsne_folder, "entire_tsne_02.png")
        plt.savefig(save_path)
        plt.close()

    def confidence_ellipse(self, x, y, ax, n_std=3.0, facecolor='none', **kwargs):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*.

        Parameters
        ----------
        x, y : array-like, shape (n, )
            Input data.

        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.

        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.

        Returns
        -------
        matplotlib.patches.Ellipse

        Other parameters
        ----------------
        kwargs : `~matplotlib.patches.Patch` properties
        """
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0),
                          width=ell_radius_x * 2,
                          height=ell_radius_y * 2,
                          facecolor=facecolor,
                          **kwargs)

        # Calculating the stdandard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)

        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)

        return ax.add_patch(ellipse)

    def visualize(self, ground_truths,
                  predictions_mean, predictions_var,
                  frame_length, image_save_path):
        step_size = int(math.ceil(float(frame_length) / float(len(predictions_mean))))

        plot_axis = 0
        fig, axs = plt.subplots(3, 1, sharex=True, tight_layout=True,
                                figsize=(14, 7))

        legend_flags = [False] * (self.dataset.number_of_classes - 1)
        for ground_truth in ground_truths:
            target_class = ground_truth[0]
            start_index = ground_truth[1]
            end_index = ground_truth[2]

            x = np.arange(start_index, end_index + 1, 1)
            y = np.array([1.0] * len(x))
            y_0 = np.array([0.0] * len(x))

            if not legend_flags[target_class - 1]:
                axs[plot_axis].fill_between(x, y, y_0,
                                            color=self.color_maps.to_rgba(target_class),
                                            label=self.dataset.label_dic[
                                                str(target_class)])
                legend_flags[target_class - 1] = True
            else:
                axs[plot_axis].fill_between(x, y, y_0,
                                            color=self.color_maps.to_rgba(target_class))
        axs[plot_axis].set_title("Ground-truths")
        axs[plot_axis].legend()
        plot_axis += 1

        top_k = 4
        top_k_classes = \
            np.sum(predictions_mean, axis=0)
        top_k_classes = np.argsort(top_k_classes)[::-1][:top_k]

        for k in range(top_k):
            x = np.arange(1, frame_length + 1, 1)
            y = np.repeat(predictions_mean[..., top_k_classes[k]],
                          step_size)[:frame_length]
            axs[plot_axis].plot(x, y,
                                color=self.color_maps.to_rgba(top_k_classes[k]),
                                label=self.dataset.label_dic[str(top_k_classes[k])])
        axs[plot_axis].legend()
        axs[plot_axis].set_title("Spotting Predictions")
        plot_axis += 1

        for k in range(top_k):
            x = np.arange(1, frame_length + 1, 1)
            y = np.repeat(predictions_var[..., top_k_classes[k]],
                          step_size)[:frame_length]
            axs[plot_axis].plot(x, y,
                                color=self.color_maps.to_rgba(top_k_classes[k]),
                                label=self.dataset.label_dic[str(top_k_classes[k])])
        axs[plot_axis].legend()
        axs[plot_axis].set_title("Spotting Prediction Variances")

        plt.savefig(image_save_path)
        plt.close(fig)

    def generate_boundaries(self):
        self.is_server = False
        self.dataset_name = "thumos14"
        self.data_type = "images"
        self.flow_type = "tvl1"
        self.temporal_width = 64
        self.feature_interval = 2

        self.dataset = self.Dataset(self)

        self.feature_folder = os.path.join(self.dataset.dataset_folder,
                                           "SpottingNetwork_Data_thumos14_0925",
                                           "features")

        self.foreground_threshold = 0.1
        self.background_threshold = 0.1
        self.entropy_threshold = 0.1
        self.expansion_ratio = 1.25

        self.maximum_duration = 500 // self.feature_interval

        start_boundary_tp = 0.0
        start_boundary_fp = 0.0
        start_boundary_fn = 0.0

        end_boundary_tp = 0.0
        end_boundary_fp = 0.0
        end_boundary_fn = 0.0

        instance_tp = 0.0
        instance_fp = 0.0
        instance_fn = 0.0

        mapping = \
            lambda x: int(round((float(x * self.feature_interval) + float((x + 1) * self.feature_interval)) / 2.0))

        identities = self.dataset.meta_dic["database"].keys()
        for identity in tqdm(identities, desc="Generating Boundaries"):
            predictions = np.load(os.path.join(self.feature_folder, identity,
                                               "{}_predictions.npy".format(identity)))
            backgrounds = predictions[:, 0]
            entropy = -np.sum(np.log(predictions) * predictions, axis=-1)

            frame_length = len(glob.glob(os.path.join(self.dataset.frames_folder, identity, "images", "*")))
            annotations = self.dataset.meta_dic["database"][identity]["annotations"]
            # C*, f_s, f_e
            ground_truths = list()
            for annotation in annotations:
                target = self.dataset.category_dic[annotation["label"].replace(" ", "_")]
                segment = annotation["segment"]

                start_index = max(1, int(math.floor(segment[0] * self.dataset.video_fps)))
                end_index = min(int(math.ceil(segment[1] * self.dataset.video_fps)), frame_length)

                if end_index - start_index + 1 >= 1:
                    ground_truths.append((target, start_index, end_index))

            foreground_peaks = list()
            for class_idx in range(self.dataset.number_of_classes - 1):
                class_foreground_peaks = self.get_peaks(predictions[:, class_idx], self.foreground_threshold)
                if len(class_foreground_peaks):
                    foreground_peaks.append([class_idx, class_foreground_peaks])
            background_peaks = self.get_peaks(backgrounds, self.background_threshold, wall=True)
            entropy_peaks = self.get_peaks(entropy, self.entropy_threshold, wall=True)

            for foreground_peak in foreground_peaks:
                nearest_background = None

                nearest_gt_instance = None
                min_distance = float("inf")
                for ground_truth in ground_truths:
                    center_idx = ground_truth[1] + ground_truth[2]
                    distance = np.abs(center_idx - mapping(foreground_peak))
                    if distance <= min_distance:
                        min_distance = distance
                        nearest_gt_instance = min_distance

    def get_peaks(self, sequence, threshold, wall=True):
        peaks = list()

        for idx in range(1, len(sequence) - 1, 1):
            if sequence[idx] >= threshold:
                if sequence[idx - 1] <= sequence[idx] and sequence[idx] >= sequence[idx + 1]:
                    peaks.append([idx, idx, 1])

        if wall:
            if sequence[0] >= threshold and sequence[0] >= sequence[1]:
                peaks.insert(0, [0, 0, 1])

            if sequence[-1] >= threshold and sequence[-1] >= sequence[-2]:
                peaks.insert(len(sequence), [len(sequence) - 1, len(sequence) - 1, 1])

        return peaks

    def get_segments(self, sequence, threshold):
        segments = list()

        thresholded_sequence = np.greater_equal(sequence, threshold).astype(np.int64)

        run = {"class_number": thresholded_sequence[0], "frames": []}
        slices = [run]
        expect = None
        index = 1
        for target in thresholded_sequence.tolist():
            if (target == expect) or (expect is None):
                run["frames"].append(index)
            else:
                run = {"class_number": target, "frames": [index]}
                slices.append(run)
            expect = target
            index += 1

        for slice in slices:
            if slice["class_number"] >= 1:
                segments.append([slice["frames"][0],
                                 slice["frames"][-1],
                                 slice["frames"][-1] - slice["frames"][0] + 1])

        return segments

    def generate_regions(self):
        self.is_server = False
        self.dataset_name = "activitynet"
        self.data_type = "images"
        self.flow_type = "tvl1"
        self.temporal_width = 64
        self.feature_interval = 2

        self.dataset = self.Dataset(self)

        if self.dataset_name == "thumos14":
            self.data_folder = os.path.join(self.dataset.dataset_folder,
                                               "SpottingNetwork_Data_{}_0925".format(self.dataset_name))
        else:
            self.data_folder = os.path.join(self.dataset.dataset_folder,
                                            "SpottingNetwork_Data_{}_1015".format(self.dataset_name))
        self.feature_folder = os.path.join(self.data_folder, "features")
        self.region_folder = os.path.join(self.data_folder, "regions")
        if not os.path.exists(self.region_folder):
            try:
                os.mkdir(self.region_folder)
            except OSError:
                pass

        self.peak_threshold = 0.1
        self.region_thresholds = np.arange(0.1, 1.0, 0.1)
        self.region_nms_threshold = 0.75
        self.smoothing = False

        self.region_save_path = os.path.join(self.region_folder, "regions.json")
        self.region_result_save_path = os.path.join(self.region_folder, "region_results.txt")

        precision_tp = 0.0
        precision_fp = 0.0

        recall_tp = 0.0
        n_gts = 0.0

        overlap = 0.0

        mapping = \
            lambda x: int(round((float(x * self.feature_interval) + float((x + 1) * self.feature_interval)) / 2.0))

        region_dict = dict()
        identities = self.dataset.meta_dic["database"].keys()
        for idx, identity in enumerate(identities):
            predictions_path = os.path.join(self.feature_folder, identity,
                                            "{}_predictions.npy".format(identity))

            if not os.path.exists(predictions_path):
                print("Generating regions ... {} {:5d}/{:5d} {:6.2f}% PASS :)".format(identity, idx + 1,
                                                                                      len(identities),
                                                                                      float(idx + 1) /
                                                                                      float(len(identities)) *
                                                                                      100.0))
                continue

            predictions = np.load(predictions_path)

            frame_length = len(glob.glob(os.path.join(self.dataset.frames_folder, identity, "images", "*")))
            annotations = self.dataset.meta_dic["database"][identity]["annotations"]
            # C*, f_s, f_e
            ground_truths = list()
            for annotation in annotations:
                target = self.dataset.category_dic[annotation["label"].replace(" ", "_")]
                segment = annotation["segment"]

                start_index = max(1, int(math.floor(segment[0] * self.dataset.video_fps)))
                end_index = min(int(math.ceil(segment[1] * self.dataset.video_fps)), frame_length)

                if end_index - start_index + 1 >= 1:
                    ground_truths.append((target, start_index, end_index))

            regions = list()
            for class_idx in range(1, self.dataset.number_of_classes, 1):
                if self.smoothing:
                    classwise_sequence = gaussian_filter1d(predictions[:, class_idx], sigma=1, mode="nearest")
                else:
                    classwise_sequence = predictions[:, class_idx]
                classwise_regions = self.get_peaks(classwise_sequence, self.peak_threshold)

                for region_threshold in self.region_thresholds:
                    classwise_regions += self.get_segments(classwise_sequence, region_threshold)

                if len(classwise_regions):
                    classwise_regions = self.region_nms(classwise_regions, threshold=self.region_nms_threshold)
                    classwise_regions = classwise_regions[:, :-1].tolist()
                    classwise_regions.sort(key=lambda region:region[0])
                    regions.append([class_idx, classwise_regions])

            region_dict[identity] = regions

            if self.dataset.meta_dic["database"][identity]["subset"] == "validation":
                gt_flags = np.zeros(shape=(len(ground_truths)), dtype=np.float32)
                for class_idx, classwise_regions in regions:
                    for region in classwise_regions:
                        region_start_idx = mapping(region[0])
                        region_end_idx = mapping(region[1])
                        region_center_idx = int(round((region_start_idx + region_end_idx) / 2.0))

                        nearest_gt_instance = None
                        min_distance = float("inf")
                        for gt_idx, ground_truth in enumerate(ground_truths):
                            gt_center_idx = int(round((ground_truth[1] + ground_truth[2]) / 2.0))
                            distance = np.abs(gt_center_idx - region_center_idx)
                            if distance <= min_distance:
                                min_distance = distance
                                nearest_gt_instance = [gt_idx, ground_truth]

                        nearest_gt_start_idx = nearest_gt_instance[1][1]
                        nearest_gt_end_idx = nearest_gt_instance[1][2]

                        if region_start_idx >= nearest_gt_start_idx and region_end_idx <= nearest_gt_end_idx:
                            gt_flags[nearest_gt_instance[0]] = 1.0
                            precision_tp += 1.0
                            union = max(region_end_idx, nearest_gt_end_idx) - min(region_start_idx, nearest_gt_start_idx) + 1
                            intersection = \
                                min(region_end_idx, nearest_gt_end_idx) - max(region_start_idx, nearest_gt_start_idx) + 1
                            this_overlap = float(intersection) / float(union)
                            overlap += this_overlap
                        else:
                            precision_fp += 1.0

                n_gts += float(len(ground_truths))
                recall_tp += np.sum(gt_flags)

            print("Generating regions ... {} {:5d}/{:5d} {:6.2f}%".format(identity, idx + 1, len(identities),
                                                                          float(idx + 1) /
                                                                          float(len(identities)) * 100.0))

        precision = precision_tp / (precision_tp + precision_fp)
        recall = recall_tp / n_gts
        overlap = overlap / precision_tp

        with open(self.region_save_path, "w") as fp:
            json.dump(region_dict, fp, indent=4, sort_keys=True)

        result_string = "Precision: {:.5f}\n".format(precision)
        result_string += "Recall: {:.5f}\n".format(recall)
        result_string += "Overlap: {:.5f}".format(overlap)

        with open(self.region_result_save_path, "w") as fp:
            fp.write(result_string)

        print(result_string)

    def region_nms(self, regions, threshold=0.65):

        regions = np.copy(sorted(regions, key=lambda x:x[-1], reverse=True))

        # if there are no boxes, return an empty list
        if len(regions) == 0:
            return []

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = regions[:, 0]
        x2 = regions[:, 1]

        # compute the area of the bounding boxes and grab the indexes to sort
        area = (x2 - x1 + 1)

        idxs = np.arange(len(regions))

        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the index value
            # to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of the bounding
            # box and the smallest (x, y) coordinates for the end of the bounding
            # box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)

            # compute the ratio of overlap
            overlap = w / area[idxs[:last]]

            # delete all indexes from the index list that have overlap greater
            # than the provided overlap threshold
            idxs = np.delete(idxs,
                             np.concatenate(([last],
                                             np.where(overlap > threshold)[0])))

        # return only the bounding boxes that were picked
        return regions[pick]

    class Dataset():

        def __init__(self, networks):
            self.root_path = os.path.abspath("..")
            self.video_fps = 25.0
            self.networks = networks

            self.meta_folder = os.path.join(self.root_path, "meta")
            if self.networks.dataset_name == "thumos14":
                self.dataset_folder = os.path.join("/mnt/hdd0/THUMOS14")
                self.target_path = os.path.join(self.meta_folder, "thumos14.json")
                self.class_label_path = os.path.join(self.meta_folder, "thumos14_classes.txt")
            elif self.networks.dataset_name == "activitynet":
                self.dataset_folder = os.path.join("/mnt/hdd0/ActivityNet/v1.3")
                self.target_path = os.path.join(self.meta_folder, "activity_net.v1.3.min.json")
                self.class_label_path = os.path.join(self.meta_folder, "activitynet_classes.txt")
            elif self.networks.dataset_name == "hacs":
                self.dataset_folder = os.path.join("/mnt/hdd0/HACS")
                self.target_path = os.path.join(self.meta_folder, "HACS_segments_v1.1.json")
                self.class_label_path = os.path.join(self.meta_folder, "HACS_classes.txt")

            self.frames_folder = os.path.join(self.dataset_folder, "frames")
            self.training_data_folder = os.path.join(self.dataset_folder, "training")
            self.validation_data_folder = os.path.join(self.dataset_folder, "validation")
            self.testing_data_folder = os.path.join(self.dataset_folder, "testing")

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
                if self.dataset.networks.dataset_name == "thumos14":
                    if self.dataset.video_fps >= 25.0:
                        json_file_path = os.path.join(self.dataset.meta_folder, "thumos14_train_data.json")
                    else:
                        json_file_path = os.path.join(self.dataset.meta_folder,
                                                      "thumos14_{}_train_data.json".format(int(self.dataset.video_fps)))
                elif self.dataset.networks.dataset_name == "activitynet":
                    if self.dataset.video_fps >= 25.0:
                        json_file_path = os.path.join(self.dataset.meta_folder, "activitynet_train_data.json")
                    else:
                        json_file_path = os.path.join(self.dataset.meta_folder,
                                                      "activitynet_{}_train_data.json".format(
                                                          int(self.dataset.video_fps)))
                else:
                    if self.dataset.video_fps >= 25.0:
                        json_file_path = os.path.join(self.dataset.meta_folder, "hacs_train_data.json")
                    else:
                        json_file_path = os.path.join(self.dataset.meta_folder,
                                                      "hacs_{}_train_data.json".format(
                                                          int(self.dataset.video_fps)))

                if os.path.exists(json_file_path):
                    with open(json_file_path, "r") as fp:
                        tf_data = json.load(fp)
                else:
                    print("There is no json file. Make the json file")
                    training_videos = glob.glob(os.path.join(self.dataset.training_data_folder, "*"))
                    tf_data = list()
                    for training_video in training_videos:
                        if self.dataset.video_fps == -1:
                            video_cap = cv2.VideoCapture(training_video)
                            video_fps = video_cap.get(cv2.CAP_PROP_FPS)
                        else:
                            video_fps = self.dataset.video_fps

                        identity = training_video.split("/")[-1].split(".")[-2]
                        frames = len(glob.glob(os.path.join(self.dataset.frames_folder, identity, "images", "*")))

                        annotations = self.dataset.meta_dic["database"][identity]["annotations"]

                        if not frames:
                            continue

                        segments_string = ""
                        for annotation in annotations:
                            target = self.dataset.category_dic[annotation["label"].replace(" ", "_")]
                            segment = annotation["segment"]

                            start_index = max(1, int(math.floor(segment[0] * video_fps)))
                            end_index = min(int(math.ceil(segment[1] * video_fps)), frames)

                            if end_index - start_index + 1 >= 1:
                                segments_string += "{} {} {} ".format(target, start_index, end_index)

                        if len(segments_string):
                            segments_string = segments_string[:-1]

                            video = "{} {} {}".format(identity, frames, segments_string)
                            tf_datum = video
                            tf_data.append(tf_datum)

                        print("VIDEO {}: {:05d}/{:05d} Done".format(identity, training_videos.index(training_video),
                                                                    len(training_videos)))

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
                                                             [video], [tf.float32, tf.int64]),
                                                  num_parallel_calls=self.dataset.networks.num_workers)
                train_dataset = train_dataset.batch(batch_size=batch_size,
                                                    drop_remainder=True)
                train_dataset = train_dataset.prefetch(5)
                self.tf_dataset = train_dataset

            def sample(self, video):
                splits = video.decode().split(" ")

                identity = splits[0]
                frame_length = int(splits[1])
                segment_strings = splits[2:]

                background_segments = list()
                foreground_segments = list()
                start_boundary_segments = list()
                end_boundary_segments = list()

                previous_end_index = 0
                for segment_index in range(len(segment_strings) // 3):
                    target = int(segment_strings[segment_index * 3])
                    start_index = int(segment_strings[segment_index * 3 + 1])
                    end_index = int(segment_strings[segment_index * 3 + 2])

                    # boundary segments
                    # note that we add boundary segments as the possible starting frame indices.
                    # we can sample a training sequence by beginning
                    # from one of the frame indices specified in an interval

                    # start boundary segment
                    start_boundary_start_index = \
                        max(start_index - self.dataset.networks.temporal_width + 1, 1)

                    start_boundary_end_index = start_index - 1

                    if start_boundary_end_index - start_boundary_start_index >= 0:
                        start_boundary_segments.append([start_boundary_start_index, start_boundary_end_index,
                                                        ["start", start_index, target]])

                    # end boundary segment
                    end_boundary_start_index = \
                        max(max(end_index - self.dataset.networks.temporal_width + 2, 1),
                            start_index)

                    end_boundary_end_index = end_index

                    if end_boundary_end_index - end_boundary_start_index >= 0:
                        end_boundary_segments.append([end_boundary_start_index, end_boundary_end_index,
                                                      ["end", end_index, target]])

                    # background segment
                    background_start_index = previous_end_index + 1
                    background_end_index = max(start_index - 1, 1)

                    if background_end_index - background_start_index + 1 >= 1:
                        background_segments.append([background_start_index, background_end_index])

                    # foreground segment
                    if end_index - start_index + 1 >= 1:
                        foreground_segments.append([start_index, end_index, target])

                    # if the foreground segment is the last one, add the final background segment
                    if segment_index == len(segment_strings) / 3 - 1:
                        background_start_index = min(end_index + 1, frame_length)
                        background_end_index = frame_length

                        if background_end_index - background_start_index + 1 >= 1:
                            background_segments.append([background_start_index, background_end_index])

                    previous_end_index = end_index

                target_type = np.random.choice(["Background", "Foreground"], 1,
                                               p=[0.5, 0.5])
                if target_type == "Background" and len(background_segments):
                    target = 0
                    background_segment = random.choice(background_segments)

                    target_frames = list()
                    segment_frame_length = background_segment[1] - background_segment[0] + 1
                    if segment_frame_length >= self.dataset.networks.temporal_width:
                        background_start_index = random.choice(list(range(background_segment[0],
                                                                          background_segment[1] -
                                                                          self.dataset.networks.temporal_width + 1 + 1,
                                                                          1)))
                        background_end_index = background_start_index + self.dataset.networks.temporal_width - 1
                        target_frames = list(range(background_start_index, background_end_index + 1, 1))
                    else:
                        frame_index = 0
                        while True:
                            sampled_frame = background_segment[0] + (frame_index % segment_frame_length)
                            target_frames.append(sampled_frame)
                            frame_index += 1
                            if frame_index >= self.dataset.networks.temporal_width:
                                break
                else:
                    foreground_segment = random.choice(foreground_segments)
                    target = foreground_segment[2] - 1

                    target_frames = list()
                    segment_frame_length = foreground_segment[1] - foreground_segment[0] + 1
                    if segment_frame_length >= self.dataset.networks.temporal_width:
                        foreground_start_index = random.choice(list(range(foreground_segment[0],
                                                                          foreground_segment[1] -
                                                                          self.dataset.networks.temporal_width + 1 + 1,
                                                                          1)))
                        foreground_end_index = foreground_start_index + self.dataset.networks.temporal_width - 1
                        target_frames = list(range(foreground_start_index, foreground_end_index + 1, 1))
                    else:
                        frame_index = 0
                        while True:
                            sampled_frame = foreground_segment[0] + (frame_index % segment_frame_length)
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
                target = np.array(target, dtype=np.int64)

                return frame_vectors, target

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
                if self.dataset.networks.dataset_name == "thumos14":
                    if self.dataset.video_fps >= 25.0:
                        json_file_path = os.path.join(self.dataset.meta_folder, "thumos14_validation_data.json")
                    else:
                        json_file_path = os.path.join(self.dataset.meta_folder,
                                                      "thumos14_{}_validation_data.json".format(
                                                          int(self.dataset.video_fps)))
                elif self.dataset.networks.dataset_name == "activitynet":
                    if self.dataset.video_fps >= 25.0:
                        json_file_path = os.path.join(self.dataset.meta_folder, "activitynet_validation_data.json")
                    else:
                        json_file_path = os.path.join(self.dataset.meta_folder,
                                                      "activitynet_{}_validation_data.json".format(
                                                          int(self.dataset.video_fps)))
                else:
                    if self.dataset.video_fps >= 25.0:
                        json_file_path = os.path.join(self.dataset.meta_folder, "hacs_validation_data.json")
                    else:
                        json_file_path = os.path.join(self.dataset.meta_folder,
                                                      "hacs_{}_validation_data.json".format(
                                                          int(self.dataset.video_fps)))

                if os.path.exists(json_file_path):
                    with open(json_file_path, "r") as fp:
                        tf_data = json.load(fp)
                else:
                    print("There is no json file. Make the json file")
                    validation_videos = glob.glob(os.path.join(self.dataset.validation_data_folder, "*"))
                    tf_data = list()
                    for validation_video in validation_videos:
                        if self.dataset.video_fps == -1:
                            video_cap = cv2.VideoCapture(validation_video)
                            video_fps = video_cap.get(cv2.CAP_PROP_FPS)
                        else:
                            video_fps = self.dataset.video_fps

                        identity = validation_video.split("/")[-1].split(".")[-2]
                        frames = len(
                            glob.glob(os.path.join(self.dataset.frames_folder, identity, "images", "*")))

                        annotations = self.dataset.meta_dic["database"][identity]["annotations"]

                        if not frames:
                            continue

                        segments_string = ""
                        for annotation in annotations:
                            target = self.dataset.category_dic[annotation["label"].replace(" ", "_")]
                            segment = annotation["segment"]

                            start_index = max(1, int(math.floor(segment[0] * video_fps)))
                            end_index = min(int(math.ceil(segment[1] * video_fps)), frames)

                            if end_index - start_index + 1 >= 1:
                                segments_string += "{} {} {} ".format(target, start_index, end_index)

                        if len(segments_string):
                            segments_string = segments_string[:-1]

                            video = "{} {} {}".format(identity, frames, segments_string)
                            tf_datum = video
                            tf_data.append(tf_datum)

                        print(
                            "VIDEO {}: {:05d}/{:05d} Done".format(identity,
                                                                  validation_videos.index(validation_video) + 1,
                                                                  len(validation_videos)))

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
                                                                       [video], [tf.float32, tf.int64, tf.string]),
                                                            num_parallel_calls=self.dataset.networks.num_workers)
                validation_dataset = validation_dataset.batch(batch_size)
                validation_dataset = validation_dataset.prefetch(5)
                self.tf_dataset = validation_dataset

            def sample(self, video):
                splits = video.decode().split(" ")

                identity = splits[0]
                frame_length = int(splits[1])
                segment_strings = splits[2:]

                background_segments = list()
                foreground_segments = list()
                start_boundary_segments = list()
                end_boundary_segments = list()

                previous_end_index = 0
                for segment_index in range(len(segment_strings) // 3):
                    target = int(segment_strings[segment_index * 3])
                    start_index = int(segment_strings[segment_index * 3 + 1])
                    end_index = int(segment_strings[segment_index * 3 + 2])

                    # boundary segments
                    # note that we add boundary segments as the possible starting frame indices.
                    # we can sample a training sequence by beginning
                    # from one of the frame indices specified in an interval

                    # start boundary segment
                    start_boundary_start_index = \
                        max(start_index - self.dataset.networks.temporal_width + 1, 1)

                    start_boundary_end_index = start_index - 1

                    if start_boundary_end_index - start_boundary_start_index >= 0:
                        start_boundary_segments.append([start_boundary_start_index, start_boundary_end_index,
                                                        ["start", start_index, target]])

                    # end boundary segment
                    end_boundary_start_index = \
                        max(max(end_index - self.dataset.networks.temporal_width + 2, 1),
                            start_index)

                    end_boundary_end_index = end_index

                    if end_boundary_end_index - end_boundary_start_index >= 0:
                        end_boundary_segments.append([end_boundary_start_index, end_boundary_end_index,
                                                      ["end", end_index, target]])

                    # background segment
                    background_start_index = previous_end_index + 1
                    background_end_index = max(start_index - 1, 1)

                    if background_end_index - background_start_index + 1 >= 1:
                        background_segments.append([background_start_index, background_end_index])

                    # foreground segment
                    if end_index - start_index + 1 >= 1:
                        foreground_segments.append([start_index, end_index, target])

                    # if the foreground segment is the last one, add the final background segment
                    if segment_index == len(segment_strings) / 3 - 1:
                        background_start_index = min(end_index + 1, frame_length)
                        background_end_index = frame_length

                        if background_end_index - background_start_index + 1 >= 1:
                            background_segments.append([background_start_index, background_end_index])

                    previous_end_index = end_index

                target_type = np.random.choice(["Background", "Foreground"], 1,
                                               p=[0.5, 0.5])
                if target_type == "Background" and len(background_segments):
                    target = 0
                    background_segment = random.choice(background_segments)

                    target_frames = list()
                    segment_frame_length = background_segment[1] - background_segment[0] + 1
                    if segment_frame_length >= self.dataset.networks.validation_temporal_width:
                        background_start_index = random.choice(list(range(background_segment[0],
                                                                          background_segment[1] -
                                                                          self.dataset.networks.validation_temporal_width + 1 + 1,
                                                                          1)))
                        background_end_index = background_start_index + self.dataset.networks.validation_temporal_width - 1
                        target_frames = list(range(background_start_index, background_end_index + 1, 1))
                    else:
                        frame_index = 0
                        while True:
                            sampled_frame = background_segment[0] + (frame_index % segment_frame_length)
                            target_frames.append(sampled_frame)
                            frame_index += 1
                            if frame_index >= self.dataset.networks.validation_temporal_width:
                                break
                else:
                    foreground_segment = random.choice(foreground_segments)
                    target = foreground_segment[2] - 1

                    target_frames = list()
                    segment_frame_length = foreground_segment[1] - foreground_segment[0] + 1
                    if segment_frame_length >= self.dataset.networks.validation_temporal_width:
                        foreground_start_index = random.choice(list(range(foreground_segment[0],
                                                                          foreground_segment[1] -
                                                                          self.dataset.networks.validation_temporal_width + 1 + 1,
                                                                          1)))
                        foreground_end_index = foreground_start_index + self.dataset.networks.validation_temporal_width - 1
                        target_frames = list(range(foreground_start_index, foreground_end_index + 1, 1))
                    else:
                        frame_index = 0
                        while True:
                            sampled_frame = foreground_segment[0] + (frame_index % segment_frame_length)
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

                target = np.array(target, dtype=np.int64)

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

        class Test():

            def __init__(self, dataset):
                self.dataset = dataset

                validation_videos = glob.glob(os.path.join(self.dataset.validation_data_folder, "*"))

                print("Converting Json Validation Data to Tensor Data ...")
                if self.dataset.networks.dataset_name == "thumos14":
                    if self.dataset.video_fps >= 25.0:
                        json_file_path = os.path.join(self.dataset.meta_folder, "thumos14_validation_data.json")
                    else:
                        json_file_path = os.path.join(self.dataset.meta_folder,
                                                      "thumos14_{}_validation_data.json".format(
                                                          int(self.dataset.video_fps)))
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
                    tf_data = list()
                    for validation_video in validation_videos:
                        if self.dataset.video_fps == -1:
                            video_cap = cv2.VideoCapture(validation_video)
                            video_fps = video_cap.get(cv2.CAP_PROP_FPS)
                        else:
                            video_fps = self.dataset.video_fps

                        identity = validation_video.split("/")[-1].split(".")[-2]
                        frames = len(
                            glob.glob(os.path.join(self.dataset.frames_folder, identity, "images", "*")))

                        annotations = self.dataset.meta_dic["database"][identity]["annotations"]

                        if not frames:
                            continue

                        segments_string = ""
                        for annotation in annotations:
                            target = self.dataset.category_dic[annotation["label"].replace(" ", "_")]
                            segment = annotation["segment"]

                            start_index = max(1, int(math.floor(segment[0] * video_fps)))
                            end_index = min(int(math.ceil(segment[1] * video_fps)), frames)

                            if end_index - start_index + 1 >= 1:
                                segments_string += "{} {} {} ".format(target, start_index, end_index)

                        if len(segments_string):
                            segments_string = segments_string[:-1]

                            video = "{} {} {}".format(identity, frames, segments_string)
                            tf_datum = video
                            tf_data.append(tf_datum)

                        print(
                            "VIDEO {}: {:05d}/{:05d} Done".format(identity,
                                                                  validation_videos.index(validation_video) + 1,
                                                                  len(validation_videos)))

                        with open(json_file_path, "w") as fp:
                            json.dump(tf_data, fp, indent=4, sort_keys=True)

                self.data_count = len(tf_data)

                print("Making Tensorflow Validation Dataset Object ... {} Instances".format(len(tf_data)))
                validation_dataset = tf.data.Dataset.from_tensor_slices(tf_data)
                validation_dataset = validation_dataset.shuffle(buffer_size=len(tf_data))
                validation_dataset = validation_dataset.map(lambda video:
                                                            tf.py_func(self.sampleVideos,
                                                                       [video], [tf.string, tf.int64, tf.int64,
                                                                                 tf.int64, tf.int64]),
                                                            num_parallel_calls=self.dataset.networks.num_workers)
                self.tf_dataset = validation_dataset

            def sampleVideos(self, video):
                splits = video.decode().split(" ")

                identity = splits[0]
                frame_length = int(splits[1])
                segments_splits = splits[2:]

                segments = []
                return_segments = []
                for segment_index in range(0, len(segments_splits), 3):
                    class_number = int(segments_splits[segment_index])
                    start_index = int(segments_splits[segment_index + 1])
                    end_index = int(segments_splits[segment_index + 2])
                    segment = dict()
                    segment["class_number"] = class_number
                    segment["intervals"] = [start_index, end_index]
                    segments.append(segment)
                    return_segments.append([class_number, start_index, end_index])

                if self.dataset.networks.validation_temporal_sample_size == -1:
                    frames = range(1, frame_length + 1, 1)
                else:
                    if frame_length < self.dataset.networks.validation_temporal_sample_size:
                        frames = range(1, frame_length + 1, 1)
                        frames += [-1] * (self.dataset.networks.validation_temporal_sample_size - frame_length)
                    else:
                        start_index = \
                            random.sample(
                                range(1, (frame_length - self.dataset.networks.validation_temporal_sample_size + 1) + 1,
                                      1), 1)[0]
                        frames = range(start_index, start_index + self.dataset.networks.validation_temporal_sample_size,
                                       1)

                targets = []
                for frame in frames:
                    if frame == -1:
                        target = -1
                    else:
                        target = 0
                        for segment in segments:
                            intervals = segment["intervals"]
                            if frame >= intervals[0] and frame <= intervals[1]:
                                target = segment["class_number"]

                    targets.append(target)

                one_frame = cv2.imread(os.path.join(self.dataset.frames_folder, identity, "images", "img_00001.jpg"))

                height, width, _ = one_frame.shape

                return identity, frames, targets, (width, height), return_segments

            def rgb_preprocessing(self, batch_datum):
                splits = tf.string_split([batch_datum], delimiter=" ").values

                identity = tf.cast(splits[0], tf.string)
                frame = tf.cast(splits[1], tf.string)
                target = tf.string_to_number(splits[2], tf.int64)
                width = tf.string_to_number(splits[3], tf.int64)
                height = tf.string_to_number(splits[4], tf.int64)

                image_path = tf.string_join([self.dataset.frames_folder, "/", identity, "/images/", "img_",
                                             frame, ".jpg"])

                image = tf.read_file(image_path)
                image = tf.image.decode_jpeg(image, channels=3)

                image_channels = 3

                total_crop_height = (height - self.dataset.input_size[1])
                crop_top = total_crop_height // 2
                total_crop_width = (width - self.dataset.input_size[0])
                crop_left = total_crop_width // 2
                image = tf.slice(image, [crop_top, crop_left, 0],
                                 [self.dataset.input_size[1], self.dataset.input_size[0], -1])

                image.set_shape([self.dataset.input_size[1], self.dataset.input_size[0], image_channels])

                image = tf.divide(tf.cast(image, tf.float32), 255.0)

                frame_vector = tf.multiply(tf.subtract(tf.cast(image, tf.float32), 0.5), 2.0)

                frame_target = target

                return frame_vector, frame_target

            def flow_preprocessing(self, batch_datum):
                splits = tf.string_split([batch_datum], delimiter=" ").values

                identity = tf.cast(splits[0], tf.string)
                frame = tf.cast(splits[1], tf.string)
                target = tf.string_to_number(splits[2], tf.int64)
                width = tf.string_to_number(splits[3], tf.int64)
                height = tf.string_to_number(splits[4], tf.int64)

                flow_x_path = tf.string_join([self.dataset.frames_folder, "/", identity,
                                              "/flows_{}/".format(self.dataset.networks.flow_type),
                                              "flow_x_", frame, ".jpg"])

                flow_x = tf.read_file(flow_x_path)
                flow_x = tf.image.decode_jpeg(flow_x, channels=1)

                flow_y_path = tf.string_join([self.dataset.frames_folder, "/", identity,
                                              "/flows_{}/".format(self.dataset.networks.flow_type),
                                              "flow_y_", frame, ".jpg"])

                flow_y = tf.read_file(flow_y_path)
                flow_y = tf.image.decode_jpeg(flow_y, channels=1)

                flow = tf.concat([flow_x, flow_y], axis=2)

                image_channels = 2

                total_crop_height = (height - self.dataset.input_size[1])
                crop_top = total_crop_height // 2
                total_crop_width = (width - self.dataset.input_size[0])
                crop_left = total_crop_width // 2
                flow = tf.slice(flow, [crop_top, crop_left, 0],
                                [self.dataset.input_size[1], self.dataset.input_size[0], -1])

                flow.set_shape([self.dataset.input_size[1], self.dataset.input_size[0], image_channels])

                flow = tf.divide(tf.cast(flow, tf.float32), 255.0)

                frame_vector = tf.multiply(tf.subtract(tf.cast(flow, tf.float32), 0.5), 2.0)

                frame_target = target

                return frame_vector, frame_target

        class Make():

            def __init__(self, dataset, dataset_type):
                self.dataset = dataset
                self.dataset_type = dataset_type

                tf_data = list()
                if self.dataset_type in ["training", "all"]:
                    print("Converting Json Train Data to Tensor Data ...")
                    if self.dataset.networks.dataset_name == "thumos14":
                        if self.dataset.video_fps >= 25.0:
                            json_file_path = os.path.join(self.dataset.meta_folder, "thumos14_train_data.json")
                        else:
                            json_file_path = os.path.join(self.dataset.meta_folder,
                                                          "thumos14_{}_train_data.json".format(
                                                              int(self.dataset.video_fps)))
                    elif self.dataset.networks.dataset_name == "activitynet":
                        if self.dataset.video_fps >= 25.0:
                            json_file_path = os.path.join(self.dataset.meta_folder, "activitynet_train_data.json")
                        else:
                            json_file_path = os.path.join(self.dataset.meta_folder,
                                                          "activitynet_{}_train_data.json".format(
                                                              int(self.dataset.video_fps)))
                    else:
                        if self.dataset.video_fps >= 25.0:
                            json_file_path = os.path.join(self.dataset.meta_folder, "hacs_train_data.json")
                        else:
                            json_file_path = os.path.join(self.dataset.meta_folder,
                                                          "hacs_{}_train_data.json".format(
                                                              int(self.dataset.video_fps)))

                    if os.path.exists(json_file_path):
                        with open(json_file_path, "r") as fp:
                            tf_data += json.load(fp)
                    else:
                        print("There is no json file. Make the json file")
                        training_videos = glob.glob(os.path.join(self.dataset.training_data_folder, "*"))
                        this_tf_data = list()
                        for training_video in training_videos:
                            if self.dataset.video_fps == -1:
                                video_cap = cv2.VideoCapture(training_video)
                                video_fps = video_cap.get(cv2.CAP_PROP_FPS)
                            else:
                                video_fps = self.dataset.video_fps

                            identity = training_video.split("/")[-1].split(".")[-2]
                            frames = len(glob.glob(os.path.join(self.dataset.frames_folder, identity, "images", "*")))

                            annotations = self.dataset.meta_dic["database"][identity]["annotations"]

                            if not frames:
                                continue

                            segments_string = ""
                            for annotation in annotations:
                                target = self.dataset.category_dic[annotation["label"].replace(" ", "_")]
                                segment = annotation["segment"]

                                start_index = max(1, int(math.floor(segment[0] * video_fps)))
                                end_index = min(int(math.ceil(segment[1] * video_fps)), frames)

                                if end_index - start_index + 1 >= 1:
                                    segments_string += "{} {} {} ".format(target, start_index, end_index)

                            if len(segments_string):
                                segments_string = segments_string[:-1]

                                video = "{} {} {}".format(identity, frames, segments_string)
                                tf_datum = video
                                this_tf_data.append(tf_datum)

                            print("VIDEO {}: {:05d}/{:05d} Done".format(identity, training_videos.index(training_video),
                                                                        len(training_videos)))

                        with open(json_file_path, "w") as fp:
                            json.dump(this_tf_data, fp, indent=4, sort_keys=True)

                        tf_data += this_tf_data

                if self.dataset_type in ["validation", "all"]:
                    print("Converting Json Validation Data to Tensor Data ...")
                    if self.dataset.networks.dataset_name == "thumos14":
                        if self.dataset.video_fps >= 25.0:
                            json_file_path = os.path.join(self.dataset.meta_folder, "thumos14_validation_data.json")
                        else:
                            json_file_path = os.path.join(self.dataset.meta_folder,
                                                          "thumos14_{}_validation_data.json".format(
                                                              int(self.dataset.video_fps)))
                    elif self.dataset.networks.dataset_name == "activitynet":
                        if self.dataset.video_fps >= 25.0:
                            json_file_path = os.path.join(self.dataset.meta_folder, "activitynet_validation_data.json")
                        else:
                            json_file_path = os.path.join(self.dataset.meta_folder,
                                                          "activitynet_{}_validation_data.json".format(
                                                              int(self.dataset.video_fps)))
                    else:
                        if self.dataset.video_fps >= 25.0:
                            json_file_path = os.path.join(self.dataset.meta_folder, "hacs_validation_data.json")
                        else:
                            json_file_path = os.path.join(self.dataset.meta_folder,
                                                          "hacs_{}_validation_data.json".format(
                                                              int(self.dataset.video_fps)))

                    if os.path.exists(json_file_path):
                        with open(json_file_path, "r") as fp:
                            tf_data += json.load(fp)
                    else:
                        print("There is no json file. Make the json file")
                        validation_videos = glob.glob(os.path.join(self.dataset.validation_data_folder, "*"))
                        this_tf_data = list()
                        for validation_video in validation_videos:
                            if self.dataset.video_fps == -1:
                                video_cap = cv2.VideoCapture(validation_video)
                                video_fps = video_cap.get(cv2.CAP_PROP_FPS)
                            else:
                                video_fps = self.dataset.video_fps

                            identity = validation_video.split("/")[-1].split(".")[-2]
                            frames = len(
                                glob.glob(os.path.join(self.dataset.frames_folder, identity, "images", "*")))

                            annotations = self.dataset.meta_dic["database"][identity]["annotations"]

                            if not frames:
                                continue

                            segments_string = ""
                            for annotation in annotations:
                                target = self.dataset.category_dic[annotation["label"].replace(" ", "_")]
                                segment = annotation["segment"]

                                start_index = max(1, int(math.floor(segment[0] * video_fps)))
                                end_index = min(int(math.ceil(segment[1] * video_fps)), frames)

                                if end_index - start_index + 1 >= 1:
                                    segments_string += "{} {} {} ".format(target, start_index, end_index)

                            if len(segments_string):
                                segments_string = segments_string[:-1]

                                video = "{} {} {}".format(identity, frames, segments_string)
                                tf_datum = video
                                this_tf_data.append(tf_datum)

                            print(
                                "VIDEO {}: {:05d}/{:05d} Done".format(identity,
                                                                      validation_videos.index(validation_video) + 1,
                                                                      len(validation_videos)))

                        with open(json_file_path, "w") as fp:
                            json.dump(this_tf_data, fp, indent=4, sort_keys=True)

                        tf_data += this_tf_data

                if self.dataset_type == "testing":
                    print("Converting Json Testing Data to Tensor Data ...")
                    json_file_path = \
                        os.path.join(self.dataset.meta_folder,
                                     "{}_testing_data.json".format(self.dataset.networks.dataset_name))

                    if os.path.exists(json_file_path):
                        with open(json_file_path, "r") as fp:
                            tf_data += json.load(fp)
                    else:
                        print("There is no json file. Make the json file")
                        testing_videos = glob.glob(os.path.join(self.dataset.testing_data_folder, "*"))
                        this_tf_data = list()
                        for video_idx, testing_video in enumerate(testing_videos):
                            video_fps = self.dataset.video_fps

                            identity = testing_video.split("/")[-1].split(".")[-2]
                            frames = len(
                                glob.glob(os.path.join(self.dataset.frames_folder, identity, "images", "*")))

                            if not frames:
                                continue

                            video = "{} {}".format(identity, frames)
                            tf_datum = video
                            this_tf_data.append(tf_datum)

                            print(
                                "VIDEO {}: {:05d}/{:05d} Done".format(identity,
                                                                      video_idx + 1,
                                                                      len(testing_videos)))

                        with open(json_file_path, "w") as fp:
                            json.dump(this_tf_data, fp, indent=4, sort_keys=True)

                        tf_data += this_tf_data

                self.data_count = len(tf_data)

                print("Making Tensorflow Dataset Object ... {} Instances".format(len(tf_data)))
                this_dataset = tf.data.Dataset.from_tensor_slices(tf_data)
                this_dataset = this_dataset.shuffle(buffer_size=len(tf_data))
                this_dataset = this_dataset.map(lambda video:
                                                tf.py_func(self.sampleVideos,
                                                           [video], [tf.string, tf.int64,
                                                                     tf.int64, tf.int64, tf.string]),
                                                num_parallel_calls=self.dataset.networks.num_workers)
                self.tf_dataset = this_dataset

            def sampleVideos(self, video):
                splits = video.decode().split(" ")

                identity = splits[0]
                frame_length = int(splits[1])

                if self.dataset.networks.dataset_type != "testing":
                    segments_splits = splits[2:]
                    segment_string = " ".join(segments_splits)

                    segments = []
                    return_segments = []
                    for segment_index in range(0, len(segments_splits), 3):
                        class_number = int(segments_splits[segment_index])
                        start_index = int(segments_splits[segment_index + 1])
                        end_index = int(segments_splits[segment_index + 2])
                        segment = dict()
                        segment["class_number"] = class_number
                        segment["intervals"] = [start_index, end_index]
                        segments.append(segment)
                        return_segments.append([class_number, start_index, end_index])

                frames = np.arange(1, frame_length + 1, 1)

                if self.dataset.networks.dataset_type != "testing":
                    targets = []
                    for frame in frames:
                        if frame == -1:
                            target = -1
                        else:
                            target = 0
                            for segment in segments:
                                intervals = segment["intervals"]
                                if frame >= intervals[0] and frame <= intervals[1]:
                                    target = segment["class_number"]

                        targets.append(target)
                else:
                    targets = []
                    for frame in frames:
                        if frame == -1:
                            target = -1
                        else:
                            target = 0

                        targets.append(target)

                one_frame = cv2.imread(os.path.join(self.dataset.frames_folder, identity, "images", "img_00001.jpg"))

                height, width, _ = one_frame.shape

                if self.dataset.networks.dataset_type != "testing":
                    return identity, frames, targets, (width, height), segment_string
                else:
                    return identity, frames, targets, (width, height), ""

            def rgb_preprocessing(self, batch_datum):
                splits = tf.string_split([batch_datum], delimiter=" ").values

                identity = tf.cast(splits[0], tf.string)
                frame = tf.cast(splits[1], tf.string)
                target = tf.string_to_number(splits[2], tf.int64)
                width = tf.string_to_number(splits[3], tf.int64)
                height = tf.string_to_number(splits[4], tf.int64)

                image_path = tf.string_join([self.dataset.frames_folder, "/", identity, "/images/", "img_",
                                             frame, ".jpg"])
                image_channels = 3

                image = tf.read_file(image_path)
                image = tf.image.decode_jpeg(image, channels=3)

                total_crop_height = (height - self.dataset.networks.input_size[1])
                crop_top = total_crop_height // 2
                total_crop_width = (width - self.dataset.networks.input_size[0])
                crop_left = total_crop_width // 2

                image = tf.slice(image, [crop_top, crop_left, 0],
                                 [self.dataset.networks.input_size[1],
                                  self.dataset.networks.input_size[0], -1])
                image.set_shape([self.dataset.networks.input_size[1],
                                    self.dataset.networks.input_size[0],
                                    image_channels])
                image = tf.divide(tf.cast(image, tf.float32), 255.0)

                frame_vector = tf.multiply(tf.subtract(tf.cast(image, tf.float32), 0.5), 2.0)

                frame_target = target

                return frame_vector, frame_target

            def flow_preprocessing(self, batch_datum):
                splits = tf.string_split([batch_datum], delimiter=" ").values

                identity = tf.cast(splits[0], tf.string)
                frame = tf.cast(splits[1], tf.string)
                target = tf.string_to_number(splits[2], tf.int64)
                width = tf.string_to_number(splits[3], tf.int64)
                height = tf.string_to_number(splits[4], tf.int64)

                flow_x_path = tf.string_join([self.dataset.frames_folder, "/",
                                              identity,
                                              "/flows_{}/flow_x_".format(self.dataset.networks.flow_type),
                                              frame, ".jpg"])
                flow_y_path = tf.string_join([self.dataset.frames_folder, "/",
                                              identity,
                                              "/flows_{}/flow_y_".format(self.dataset.networks.flow_type),
                                              frame, ".jpg"])
                flow_channels = 2

                flow_x = tf.read_file(flow_x_path)
                flow_x = tf.image.decode_jpeg(flow_x, channels=1)

                flow_y = tf.read_file(flow_y_path)
                flow_y = tf.image.decode_jpeg(flow_y, channels=1)

                flow = tf.concat([flow_x, flow_y], axis=2)

                total_crop_height = (height - self.dataset.networks.input_size[1])
                crop_top = total_crop_height // 2
                total_crop_width = (width - self.dataset.networks.input_size[0])
                crop_left = total_crop_width // 2
                flow = tf.slice(flow, [crop_top, crop_left, 0],
                                [self.dataset.networks.input_size[1],
                                 self.dataset.networks.input_size[0], -1])

                flow.set_shape([self.dataset.networks.input_size[1],
                                self.dataset.networks.input_size[0], flow_channels])

                flow = tf.divide(tf.cast(flow, tf.float32), 255.0)

                frame_vector = tf.multiply(tf.subtract(tf.cast(flow, tf.float32), 0.5), 2.0)
                frame_target = target

                return frame_vector, frame_target

            def rgb_preprocessing_np(self, batch_datum):
                splits = batch_datum.decode().split(" ")

                identity = splits[0]
                frame = int(splits[1])
                crop_top = int(splits[2])
                crop_left = int(splits[3])
                is_flipped = int(splits[4])

                image_path = \
                    os.path.join(self.dataset.frames_folder, identity,
                                 "images", "img_{:05d}.jpg".format(frame))
                image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

                cropped_image = \
                    image[crop_top:crop_top + self.dataset.networks.input_size[1],
                    crop_left:crop_left + self.dataset.networks.input_size[0], :]

                cropped_image = cropped_image.astype(np.float32)
                cropped_image = np.divide(cropped_image, 255.0)
                cropped_image = np.multiply(np.subtract(cropped_image, 0.5), 2.0)
                if is_flipped:
                    cropped_image = cv2.flip(cropped_image, 1)

                return cropped_image

            def flow_preprocessing_np(self, batch_datum):
                splits = batch_datum.decode().split(" ")

                identity = splits[0]
                frame = int(splits[1])
                crop_top = int(splits[2])
                crop_left = int(splits[3])
                is_flipped = int(splits[4])

                flow_x_path = \
                    os.path.join(self.dataset.frames_folder, identity,
                                 "flows_{}".format(self.dataset.networks.flow_type),
                                 "flow_x_{:05d}.jpg".format(frame))
                flow_y_path = \
                    os.path.join(self.dataset.frames_folder, identity,
                                 "flows_{}".format(self.dataset.networks.flow_type),
                                 "flow_y_{:05d}.jpg".format(frame))

                flow_x = cv2.imread(flow_x_path, cv2.IMREAD_GRAYSCALE)
                flow_y = cv2.imread(flow_y_path, cv2.IMREAD_GRAYSCALE)
                flow = np.stack([flow_x, flow_y], axis=-1)

                cropped_flow = \
                    flow[crop_top:crop_top + self.dataset.networks.input_size[1],
                    crop_left:crop_left + self.dataset.networks.input_size[0], :]

                cropped_flow = cropped_flow.astype(np.float32)
                cropped_flow = np.divide(cropped_flow, 255.0)
                cropped_flow = np.multiply(np.subtract(cropped_flow, 0.5), 2.0)
                if is_flipped:
                    cropped_flow = cv2.flip(cropped_flow, 1)

                return cropped_flow

            def fusion_preprocessing_np(self, batch_datum):
                splits = batch_datum.decode().split(" ")

                identity = splits[0]
                frame = int(splits[1])
                crop_top = int(splits[2])
                crop_left = int(splits[3])
                is_flipped = int(splits[4])

                image_path = \
                    os.path.join(self.dataset.frames_folder, identity,
                                 "images", "img_{:05d}.jpg".format(frame))
                image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

                flow_x_path = \
                    os.path.join(self.dataset.frames_folder, identity,
                                 "flows_{}".format(self.dataset.networks.flow_type),
                                 "flow_x_{:05d}.jpg".format(frame))
                flow_y_path = \
                    os.path.join(self.dataset.frames_folder, identity,
                                 "flows_{}".format(self.dataset.networks.flow_type),
                                 "flow_y_{:05d}.jpg".format(frame))

                flow_x = cv2.imread(flow_x_path, cv2.IMREAD_GRAYSCALE)
                flow_y = cv2.imread(flow_y_path, cv2.IMREAD_GRAYSCALE)
                flow = np.stack([flow_x, flow_y], axis=-1)

                cropped_image = \
                    image[crop_top:crop_top + self.dataset.networks.input_size[1],
                    crop_left:crop_left + self.dataset.networks.input_size[0], :]

                cropped_image = cropped_image.astype(np.float32)
                cropped_image = np.divide(cropped_image, 255.0)
                cropped_image = np.multiply(np.subtract(cropped_image, 0.5), 2.0)
                if is_flipped:
                    cropped_image = cv2.flip(cropped_image, 1)

                cropped_flow = \
                    flow[crop_top:crop_top + self.dataset.networks.input_size[1],
                    crop_left:crop_left + self.dataset.networks.input_size[0], :]

                cropped_flow = cropped_flow.astype(np.float32)
                cropped_flow = np.divide(cropped_flow, 255.0)
                cropped_flow = np.multiply(np.subtract(cropped_flow, 0.5), 2.0)
                if is_flipped:
                    cropped_flow = cv2.flip(cropped_flow, 1)

                return cropped_image, cropped_flow

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

    class SpottingNetwork():

        def __init__(self, networks, is_training, data_type,
                     batch_size=None, device_id=None, num_classes=None):
            self.networks = networks
            self.is_training = is_training
            self.data_type = data_type

            self.dropout_prob = 0.5
            self.weight_decay = 1.0e-7

            if batch_size is None:
                self.batch_size = \
                    self.networks.batch_size if self.is_training \
                        else self.networks.validation_batch_size
            elif batch_size == -1:
                self.batch_size = None
            else:
                self.batch_size = batch_size
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
            self.low_level_features = list()
            self.predictions = list()
            self.dense_predictions = list()

            kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
            kernel_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)
            bias_initilizer = tf.zeros_initializer()
            bias_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)

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

            self.targets = tf.placeholder(dtype=tf.int64,
                                          shape=(batch_size),
                                          name="targets")

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
                            net = I3D.build_model(inputs=inputs,
                                                  weight_decay=self.weight_decay,
                                                  end_points=self.end_points,
                                                  dtype=self.networks.dtype,
                                                  dformat=self.networks.dformat,
                                                  is_training=self.is_training,
                                                  scope=self.i3d_name)

                            mixed_3c_features = self.end_points["Mixed_3c"]
                            mixed_3c_features = tf.nn.avg_pool3d(mixed_3c_features,
                                                                 [1, 4, 1, 1, 1]
                                                                 if self.networks.dformat == "NDHWC"
                                                                 else [1, 1, 4, 1, 1],
                                                                 strides=[1, 4, 1, 1, 1]
                                                                 if self.networks.dformat == "NDHWC"
                                                                 else [1, 1, 4, 1, 1],
                                                                 padding="SAME",
                                                                 data_format=self.networks.dformat)
                            mixed_3c_features = tf.reduce_mean(mixed_3c_features,
                                                               axis=(2, 3) if self.networks.dformat == "NDHWC"
                                                               else (3, 4))
                            mixed_4f_features = self.end_points["Mixed_4f"]
                            mixed_4f_features = tf.nn.avg_pool3d(mixed_4f_features,
                                                                 [1, 2, 1, 1, 1]
                                                                 if self.networks.dformat == "NDHWC"
                                                                 else [1, 1, 2, 1, 1],
                                                                 strides=[1, 2, 1, 1, 1]
                                                                 if self.networks.dformat == "NDHWC"
                                                                 else [1, 1, 2, 1, 1],
                                                                 padding="SAME",
                                                                 data_format=self.networks.dformat)
                            mixed_4f_features = tf.reduce_mean(mixed_4f_features,
                                                               axis=(2, 3) if self.networks.dformat == "NDHWC"
                                                               else (3, 4))
                            low_level_features = tf.concat([mixed_3c_features, mixed_4f_features],
                                                           axis=-1 if self.networks.dformat == "NDHWC" else 1)
                            self.low_level_features.append(low_level_features)

                            end_point = "Logits"
                            with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
                                # For extracting Kinetics features, modify AvgPool 2x7x7 -> AvgPool 1x7x7
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
                                                                    self.num_classes - 1]
                                                             if self.networks.dformat == "NDHWC"
                                                             else [1, self.num_classes,
                                                                   1, 1, 1],
                                                             initializer=bias_initilizer,
                                                             regularizer=bias_regularizer,
                                                             trainable=self.is_training)
                                    conv = tf.nn.conv3d(net, kernel, [1, 1, 1, 1, 1], padding="SAME",
                                                        data_format=self.networks.dformat)
                                    net = tf.add(conv, biases)

                                self.dense_predictions.append(
                                    tf.nn.softmax(tf.squeeze(net,
                                                             axis=[2, 3]
                                                             if self.networks.dformat == "NDHWC"
                                                             else [3, 4]),
                                                  axis=-1 if self.networks.dformat == "NDHWC" else 1))

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
                            gradients = self.networks.optimizer.compute_gradients(loss)
                            if device_id == 0:
                                self.reg_gradients = self.networks.optimizer.compute_gradients(
                                    tf.losses.get_regularization_loss())
                            self.gradients.append(gradients)

            with tf.device("/cpu:0"):
                self.loss /= tf.constant(self.networks.num_gpus, dtype=self.networks.dtype)
                self.accuracy /= tf.constant(self.networks.num_gpus, dtype=self.networks.dtype)
                # self.features = \
                #     tf.reduce_mean(self.end_points["Mixed_5c"],
                #                    axis=(2, 3) if self.networks.dformat == "NDHWC" else (3, 4))
                self.features = \
                    tf.reduce_mean(self.end_points["Mixed_5c"],
                                   axis=(1, 2, 3) if self.networks.dformat == "NDHWC" else (2, 3, 4))
                self.features = tf.expand_dims(self.features, axis=1)
                self.low_level_features = tf.concat(self.low_level_features, axis=0)
                # self.dense_predictions = tf.concat(self.dense_predictions, axis=0)
                self.predictions = tf.concat(self.predictions, axis=0)
                self.dense_predictions = self.predictions
                self.dense_predictions = tf.expand_dims(self.dense_predictions, axis=1)
                # if self.networks.dformat == "NCDHW":
                #     self.features = tf.transpose(self.features, [0, 2, 1])
                #     self.low_level_features = tf.transpose(self.low_level_features, [0, 2, 1])
                #     self.dense_predictions = tf.transpose(self.dense_predictions, [0, 2, 1])

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

    networks.make_kinetics_features()
