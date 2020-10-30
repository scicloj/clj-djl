(ns clj-djl.training
  (:import [ai.djl.training.util ProgressBar]
           [ai.djl.training.dataset RandomAccessDataset]
           [ai.djl.training DefaultTrainingConfig TrainingConfig ParameterStore]
           [ai.djl.training.loss Loss]
           [ai.djl.training EasyTrain]
           [ai.djl.training.evaluator Accuracy]
           [ai.djl.training.listener TrainingListener LoggingTrainingListener]
           [ai.djl.ndarray.types Shape]
           [ai.djl.training.dataset Batch]
           [ai.djl.ndarray NDList]))

(defn new-progress-bar []
  (ProgressBar.))

(defn new-training-config [loss]
  (DefaultTrainingConfig. loss))

(defn new-default-training-config [loss]
  (DefaultTrainingConfig. loss))

(defn opt-initializer [config initializer]
  (.optInitializer config initializer))

(defn opt-optimizer [config optimizer]
  (.optOptimizer config optimizer))

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

(defn iter-seq
([iterable]
 (iter-seq iterable (.iterator iterable)))
([iterable iter]
 (lazy-seq
  (when (.hasNext iter)
    (cons (.next iter) (iter-seq iterable iter))))))

(defn iterate-dataset [trainer ds]
  (iter-seq (.iterateDataset trainer ds)))

(defn train-batch [trainer batch]
  (EasyTrain/trainBatch trainer batch))

(defmacro as-consumer [f]
  `(reify java.util.function.Consumer
     (accept [this arg#]
       (~f arg#))))

(defn notify-listeners [trainer callback]
  (.notifyListeners trainer (as-consumer callback)))

(defn get-manager [trainer]
  (.getManager trainer))

(defn forward [trainer input]
  (.forward trainer (NDList. input)))

(defn set-metrics [trainer metrics]
  (.setMetrics trainer metrics))

(defn parameter-store [manager copy]
  (ParameterStore. manager copy))
