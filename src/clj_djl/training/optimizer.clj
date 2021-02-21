(ns clj-djl.training.optimizer
  (:import [ai.djl.training.optimizer Optimizer]))


(defn set-learning-rate-tracker [builder tracker]
  (.setLearningRateTracker builder tracker))

(defn build [builder]
  (.build builder))

(defn sgd
  ([]
   (Optimizer/sgd))
  ([{:keys [tracker momentum weight-decay]}]
   (cond-> (Optimizer/sgd)
     tracker (.setLearningRateTracker tracker)
     momentum (.optMomentum momentum)
     weight-decay (.optWeightDecays weight-decay)
     true (.build))))

(defn adam
  ([]
   (Optimizer/adam))
  ([{:keys [tracker weight-decay beta1 beta2 epsilon]}]
   (cond-> (Optimizer/adam)
     tracker (.optLearningRateTracker tracker)
     weight-decay (.optWeightDecays weight-decay)
     beta1 (.optBeta1 beta1)
     beta2 (.optBeta2 beta2)
     epsilon (.optEpsilon epsilon)
     :always (.build))))
