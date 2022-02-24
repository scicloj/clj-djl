(ns clj-djl.training
  (:require [clj-djl.engine :as engine]
            [clj-djl.ndarray :as nd])
  (:import [ai.djl.training.util ProgressBar]
           [ai.djl.training.dataset RandomAccessDataset]
           [ai.djl.training DefaultTrainingConfig TrainingConfig ParameterStore]
           [ai.djl.training.loss Loss]
           [ai.djl.training EasyTrain]
           [ai.djl.training.evaluator Accuracy TopKAccuracy BinaryAccuracy]
           [ai.djl.training.listener TrainingListener LoggingTrainingListener]
           [ai.djl.ndarray.types Shape]
           [ai.djl.training.dataset Batch]
           [ai.djl.ndarray NDList]
           [ai.djl.engine Engine]
           [ai.djl.metric Metric Metrics]))

(defn progress-bar []
  (ProgressBar.))

(def new-progress-bar progress-bar)

(defn config [{:keys [loss devices data-manager initializer parameter optimizer evaluator listeners]}]
  (cond-> (DefaultTrainingConfig. loss)
    listeners (.addTrainingListeners (if (sequential? listeners)
                                       (into-array TrainingListener listeners)
                                       listeners))
    evaluator (.addEvaluator evaluator)
    devices (.optDevices devices)
    data-manager (.optDataManager data-manager)
    initializer (.optInitializer initializer parameter)
    optimizer (.optOptimizer optimizer)))

(def training-config config)
(def default-training-config config)

(defn new-default-training-config [loss]
  (DefaultTrainingConfig. loss))

(def new-training-config new-default-training-config)

(defn opt-initializer [config initializer parameter]
  (.optInitializer config initializer parameter))

(defn opt-optimizer [config optimizer]
  (.optOptimizer config optimizer))

(defn softmax-cross-entropy-loss []
  (Loss/softmaxCrossEntropyLoss))

(defn add-evaluator [config evaluator]
  (.addEvaluator config evaluator)
  config)

(defn get-evaluators [trainer]
  (.getEvaluators trainer))

(defn accuracy []
  (Accuracy.))

(def new-accuracy accuracy)

(defn topk-accuracy
  ([topk]
   (TopKAccuracy. topk))
  ([index topk]
   (TopKAccuracy. topk))
  ([name index topk]
   (TopKAccuracy. name topk)))

(def new-topk-accuracy topk-accuracy)

(defn binary-accuracy
  ([]
   (BinaryAccuracy.))
  ([threshold]
   (BinaryAccuracy. threshold))
  ([acc-name threshold ]
   (BinaryAccuracy. acc-name threshold ))
  ([acc-name threshold axis]
   (BinaryAccuracy. acc-name threshold axis)))

(def new-binary-accuracy binary-accuracy)

(defn add-accumulator
  "Adds an accumulator to the accuracy for the results of the evaluation with the
  given key."
  [acc key]
  (.addAccumulator acc key)
  acc)

(defn update-accumulator
  "Updates the accuracy with the given key based on a NDList of labels and
  predictions."
  [acc key label-list pred-list]
  (.updateAccumulator acc key (nd/ndlist (nd/to-type (.head label-list) :int32 false)) pred-list)
  acc)

(defn get-accumulator
  "Returns the accumulated evaluator value."
  [acc key]
  (.getAccumulator acc key))

(defn add-training-listeners [config listener]
  (.addTrainingListeners config listener)
  config)

(defn training-listeners []
  (into-array TrainingListener [(LoggingTrainingListener.)]))

(def new-default-training-listeners training-listeners)

(defn initialize
  ([trainer shapes]
   (cond
     (sequential? shapes) (.initialize trainer (into-array Shape shapes))
     (instance? Shape shapes) (.initialize trainer (into-array Shape [shapes]))
     :else (.initialize trainer (into-array Shape [shapes])))
   trainer)
  ([trainer shape & shapes]
   (.initialize trainer (into-array Shape (cons shape shapes)))
   trainer))

(defn trainer
  ([model config]
   (.newTrainer model config))
  ([{:keys [model loss devices data-manager initializer parameter optimizer listeners]}]
   (.newTrainer model
                (cond-> (DefaultTrainingConfig. loss)
                  listeners (.addTrainingListeners (if (sequential? listeners)
                                                     (into-array TrainingListener listeners)
                                                     listeners))
                  devices (.optDevices devices)
                  data-manager (.optDataManager data-manager)
                  initializer (.optInitializer initializer parameter)
                  optimizer (.optOptimizer optimizer)))))

(def new-trainer trainer)

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

(defn metrics []
  (Metrics.))

(defn set-metrics [trainer metrics]
  (.setMetrics trainer metrics)
  trainer)

(defn get-metrics
  "Get metrics from trainer, put the metrics to seq of map:
  [{\"train_progress_Accuracy\" {:timestamp 1607859588747 :value 0.68125 :unit \"count\"}}]"
  [^ai.djl.training.Trainer trainer]
  (let [metrics (.getMetrics trainer)
        metric-names (.getMetricNames metrics)]
    (into {}
          (for [n metric-names]
            {n (map (fn [m] {:timestamp (.getTimestamp m) :value (.getValue m) :unit (.getUnit m)})
                    (.getMetric metrics n))}))))

(defn parameter-store [manager copy]
  (ParameterStore. manager copy))

(defn gradient-collector
  ([]
   (engine/new-gradient-collector (engine/get-instance)))
  ([trainer]
   (.newGradientCollector trainer)))

(def new-gradient-collector gradient-collector)

(defn fit
  ([trainer nepochs train-iter]
   (EasyTrain/fit trainer nepochs train-iter nil))
  ([trainer nepochs train-iter test-iter]
   (EasyTrain/fit trainer nepochs train-iter test-iter)))

(defn train-batch [trainer batch]
  (EasyTrain/trainBatch trainer batch))

(defn validate-batch [trainer batch]
  (EasyTrain/validateBatch trainer batch))

(defn set-requires-gradient
  [ndarray requires-grad]
  (.setRequiresGradient ndarray requires-grad))

(defn get-gradient
  "Returns the gradient NDArray attached to this NDArray."
  [ndarray]
  (.getGradient ndarray))

(defn stop-gradient
  [ndarray]
  (.stopGradient ndarray))

(defn backward [gc target]
  (.backward gc target))

(defn get-devices [config]
  (vec (.getDevices config)))

(defn get-loss [trainer]
  (.getLoss trainer))

(defn get-result [trainer]
  (let [result (.getTrainingResult trainer)]
    (assoc {}
           :epochs (.getEpoch result)
           :train-accuracy (.getTrainEvaluation result "Accuracy")
           :train-loss (.getTrainLoss result)
           :validate-accuracy (.getValidateEvaluation result "Accuracy")
           :validate-loss (.getValidateLoss result))))

(def get-training-result get-result)

(defn get-model [trainer]
  (.getModel trainer))
