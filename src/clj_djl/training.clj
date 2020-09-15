(ns clj-djl.training
  (:import [ai.djl.training.util ProgressBar]
           [ai.djl.training.dataset RandomAccessDataset]
           [ai.djl.training DefaultTrainingConfig TrainingConfig]
           [ai.djl.training.loss Loss]
           [ai.djl.training EasyTrain]
           [ai.djl.training.evaluator Accuracy]
           [ai.djl.training.listener TrainingListener LoggingTrainingListener]
           [ai.djl.ndarray.types Shape]
           [ai.djl.training.dataset Batch]))

(defn new-progress-bar []
  (ProgressBar.))

(defn new-training-config [loss]
  (DefaultTrainingConfig. loss))

(defn softmax-cross-entropy-loss []
  (Loss/softmaxCrossEntropyLoss))

(defn add-evaluator [config evaluator]
  (.addEvaluator config evaluator)
  config)

(defn new-accuracy []
  (Accuracy.))

(defn add-training-listeners [config listener]
  (.addTrainingListeners config listener)
  config)

(defn new-default-training-listeners []
  (into-array TrainingListener [(LoggingTrainingListener.)]))

(defn initialize [trainer shapes]
  (.initialize trainer (into-array Shape shapes))
  trainer)

(defn new-trainer [model config]
  (.newTrainer model config))

(defn step [trainer]
  (.step trainer))

(defn close [batch]
  (.close batch))

(defn iterate-dataset [trainer ds]
  (iterator-seq (.iterateDataset trainer ds)))

(defn train-batch [trainer batch]
  (EasyTrain/trainBatch trainer batch))

(defn notify-listeners [trainer callback]
  (.notifyListeners trainer callback))
