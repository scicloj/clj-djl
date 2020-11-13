(ns clj-djl.training
  (:require [clj-djl.engine :as engine])
  (:import [ai.djl.training.util ProgressBar]
           [ai.djl.training.dataset RandomAccessDataset]
           [ai.djl.training DefaultTrainingConfig TrainingConfig ParameterStore]
           [ai.djl.training.loss Loss]
           [ai.djl.training EasyTrain]
           [ai.djl.training.evaluator Accuracy]
           [ai.djl.training.listener TrainingListener LoggingTrainingListener]
           [ai.djl.ndarray.types Shape]
           [ai.djl.training.dataset Batch]
           [ai.djl.ndarray NDList]
           [ai.djl.engine Engine]))

(defn new-progress-bar []
  (ProgressBar.))

(defn new-training-config [loss]
  (DefaultTrainingConfig. loss))

(defn default-training-config [{:keys [loss devices data-manager initializer optimizer evaluator listeners]}]
  (cond-> (DefaultTrainingConfig. loss)
    (sequential? listeners) (.addTrainingListeners (into-array TrainingListener listeners))
    evaluator (.addEvaluator evaluator)
    devices (.optDevices devices)
    data-manager (.optDataManager data-manager)
    initializer (.optInitializer initializer)
    optimizer (.optOptimizer optimizer)))

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

(defn get-evaluators [trainer]
  (.getEvaluators trainer))

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

(defn new-trainer
  ([model config]
   (.newTrainer model config))
  ([{:keys [model loss devices data-manager initializer optimizer]}]
   (.newTrainer model
                (cond-> (DefaultTrainingConfig. loss)
                  devices (.optDevices devices)
                  data-manager (.optDataManager data-manager)
                  initializer (.optInitializer initializer)
                  optimizer (.optOptimizer optimizer)))))

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
  (.setMetrics trainer metrics)
  trainer)

(defn get-metrics [trainer]
  (.getMetrics trainer))

(defn parameter-store [manager copy]
  (ParameterStore. manager copy))

(defn new-gradient-collector
  ([]
   (engine/new-gradient-collector (engine/get-instance)))
  ([trainer]
   (.newGradientCollector trainer)))

(defn fit [trainer nepochs train-iter test-iter]
  (EasyTrain/fit trainer nepochs train-iter test-iter))


(defn gradient-collector []
  (-> (Engine/getInstance) (.newGradientCollector)))

(defn attach-gradient
  "Attaches a gradient NDArray to this NDArray and marks it so
  GradientCollector.backward(NDArray) can compute the gradient with respect to it."
  [ndarray]
  (.attachGradient ndarray))

(defn get-gradient
  "Returns the gradient NDArray attached to this NDArray."
  [ndarray]
  (.getGradient ndarray))

(defn backward [gc target]
  (.backward gc target))

(defn get-devices [config]
  (vec (.getDevices config)))

(defn get-loss [trainer]
  (.getLoss trainer))

(defn get-training-result [trainer]
  (let [result (.getTrainingResult trainer)]
    (assoc {}
           :epoch (.getEpoch result)
           :train-loss (.getTrainLoss result)
           :validate-loss (.getValidateLoss result))))
