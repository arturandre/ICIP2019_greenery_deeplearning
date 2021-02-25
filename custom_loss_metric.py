import tensorflow as tf
from keras import backend as K
from keras.utils import to_categorical
import numpy as np

def automatic_weighted_categorical_crossentropy(interestClasses, ignoredLabels=None):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        interestClasses: id's used for classes (e.g. [1,2,3,32])
        ignoredLabels: list with id's that should not be accounted
    
    Usage:
        ignoredLabels = np.asarray([31]) # Pixels labeled as 31 will be regarded as not annotated
        loss = automatic_weighted_categorical_crossentropy([1,2,3,37], [31])
        model.compile(loss=loss,optimizer='adam')
    """
    
    def loss(y_true, y_pred):

        # Labels from one-hot encode to id
        labels = tf.argmax(y_true, axis=-1)
        with tf.Session() as sess:
            flattenedLabels = K.reshape(K.flatten(labels), [-1])
            uniqueLabels, _, counts = tf.unique_with_counts(flattenedLabels, out_idx=tf.int64)
            indices = tf.nn.top_k(uniqueLabels, k=tf.size(uniqueLabels)).indices
            
            indices = tf.cast(indices, tf.int64)
            uniqueLabels = K.gather(uniqueLabels, indices[::-1])
            counts = K.gather(counts, indices[::-1])
            
            nonlocal interestClasses
            nonlocal ignoredLabels
            
            hotEncodeVecLen = np.max(np.concatenate((interestClasses, ignoredLabels or None)))+1
            
            paddedCounts_z = tf.zeros(hotEncodeVecLen, dtype = tf.int64)
            paddedCounts = tf.scatter_add(tf.Variable(paddedCounts_z), uniqueLabels, counts)
            
            if not ignoredLabels is None:
                ignoredHotEncoded = [to_categorical(ignored_label, num_classes = hotEncodeVecLen) for ignored_label in ignoredLabels]
                samplesHotEncoded = K.cast(tf.logical_not(K.any(K.stack(ignoredHotEncoded, axis=0), axis=0)), 'int64')
                paddedCounts = paddedCounts * len(interestClasses)
            
            countsSum = K.sum(paddedCounts)
            
            """
            paddedCounts shall be a vector with countage of each label in each of its positions
            positions with zeros or have been ignored or are inexistent in this sample/batch
            since we're weighting based on label's frequency per sample/batch then we should count how many nonzero position
            are there and use it as the Number of Classes (present in the current sample image)
            """
            classDivisor = paddedCounts * tf.count_nonzero(paddedCounts)
            
            classDivide = countsSum / classDivisor
            class_weights = classDivide
            
            #Set inf and NaN to 0
            #https://stackoverflow.com/a/38527873/3562468
            class_weights = tf.where(tf.is_nan(class_weights), tf.ones_like(class_weights) * 0, class_weights)
            class_weights = tf.where(tf.is_inf(class_weights), tf.ones_like(class_weights) * 0, class_weights)

            weights = tf.cast(class_weights, tf.float32)

            # scale predictions so that the class probs of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1)
            
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)
            return loss
    
    return loss

# Found at: https://stackoverflow.com/a/41717938/3562468
def single_class_precision(interesting_class_id):
    def prec(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        
        # Replace class_id_preds with class_id_true for recall here
        predict_positive_mask    = K.cast(K.equal(class_id_preds, interesting_class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true , class_id_preds), 'int32') * predict_positive_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return prec

# Adapted by Artur André A.M. Oliveira:

# Labels such as "void" usually are non annotated samples. Ignored labels are considered as samples
# that should not be counted neither as Positive, nor Negative predictions, thus they don't have
# any influence over the accuracy account;
# Accuracy = (TP+TN)/(TP+TN+FP+FN)
def single_class_accuracy(interesting_class_id, ignored_labels):
    def accuracy(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        
        class_sample_positive = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')  #Only interesting_class_id
        class_pred_positive   = K.cast(K.equal(class_id_preds, interesting_class_id), 'int32') #Only interesting_class_id
        
        if ignored_labels is None:
            # gt and prediction are equal only when they agree (True/True -> TP | False/False -> TN)
            # gt and prediction are different only when they disagree (True/False -> FP | False/True -> FN)
            class_acc_tensor = K.cast(K.equal(class_sample_positive , class_pred_positive), 'int32')
            
            # This division represents agreement over total = (TP+TN)/(TP+TN+FP+FN)
            class_acc = K.sum(class_acc_tensor) / tf.size(class_id_true) # 
        else:
            if interesting_class_id in ignored_labels:
                raise ValueError(f'Interesting class ({interesting_class_id}) should not be in the ignored labels ({ignored_labels}).')
            ignore_masks     = [K.equal(class_id_true, ignored_label) for ignored_label in ignored_labels]
            sample_mask      = K.cast(tf.logical_not(K.any(K.stack(ignore_masks, axis=0), axis=0)), 'int32')
            class_acc_tensor = K.cast(K.equal(class_sample_positive , class_pred_positive), 'int32') * sample_mask
        
        class_acc = K.sum(class_acc_tensor) / K.sum(sample_mask)
        return class_acc
    return accuracy

def single_class_IoU(interesting_class_id, ignored_labels):
    def IoU(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        
        class_sample_positive = K.equal(class_id_true, interesting_class_id)  #Positive Labels
        class_pred_positive   = K.equal(class_id_preds, interesting_class_id) #Positive Prediction
        
        if ignored_labels is None:
            #True Positives (positive agreement)
            class_tp_tensor = K.cast(tf.logical_and(class_sample_positive , class_pred_positive), 'int32')
            
            #False Positive + False negative (disagreement)
            class_false_tensor = K.cast(tf.logical_not(K.equal(class_sample_positive , class_pred_positive)), 'int32')
        else:
            if interesting_class_id in ignored_labels:
                raise ValueError(f'Interesting class ({interesting_class_id}) should not be in the ignored labels ({ignored_labels}).')
            ignore_masks     = [K.equal(class_id_true, ignored_label) for ignored_label in ignored_labels]
            sample_mask      = K.cast(tf.logical_not(K.any(K.stack(ignore_masks, axis=0), axis=0)), 'int32')
            
            class_tp_tensor = K.cast(tf.logical_and(class_sample_positive , class_pred_positive), 'int32') * sample_mask
            class_false_tensor = K.cast(tf.logical_not(K.equal(class_sample_positive , class_pred_positive)), 'int32') * sample_mask
            
        # This division represents (positive) agreement over disagreement = (TP)/(TP+FP+FN)
        class_iou = K.sum(class_tp_tensor) / (K.sum(class_tp_tensor) + K.sum(class_false_tensor))
        return class_iou
    return IoU

# Adapted by Artur André A.M. Oliveira:

# Labels such as "void" usually are non annotated samples. Ignored labels are considered as samples
# that should not be counted neither as Positive, nor Negative predictions, thus they don't have
# any influence over the IoU account;
#IoU = TP/(TP+FP+FN) -> With multiple classes it's not very clear what a True Negative means.
#In this case TP can be considered as agreement between a prediction and the true label
#FP + FN can be considered as disagreement between a prediction and the true label
#So IoU is effectively a metric based upon agreement over total -> agreements/(agreements+disagreements)
def multi_class_iou(ignored_labels=None):
    def IoU(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        
        if ignored_labels is None:
            class_acc_tensor = K.cast(K.equal(class_id_true , class_id_preds), 'int32')
            class_acc = K.sum(class_acc_tensor) / tf.size(class_id_true)
        else:
            ignore_masks     = [K.equal(class_id_true, ignored_label) for ignored_label in ignored_labels]
        
            sample_mask      = K.cast(tf.logical_not(K.any(K.stack(ignore_masks, axis=0), axis=0)), 'int32') 

            class_acc_tensor = K.cast(K.equal(class_id_true , class_id_preds), 'int32') * sample_mask

            class_acc = K.sum(class_acc_tensor) / K.sum(sample_mask)
        return class_acc
    return IoU