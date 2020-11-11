(ns clj-djl.training.optimizer
  (:import [ai.djl.training.optimizer Optimizer]))


(defn set-learning-rate-tracker [builder tracker]
  (.setLearningRateTracker builder tracker))

(defn build [builder]
  (.build builder))

(defn sgd
  ([]
   (Optimizer/sgd))
  ([{:keys [tracker momentum]}]
   (cond-> (Optimizer/sgd)
     tracker (.setLearningRateTracker tracker)
     momentum (.optMomentum momentum)
     true (.build))))
