(ns clj-djl.training.optimizer
  (:import [ai.djl.training.optimizer Optimizer]))

(defn sgd []
  (Optimizer/sgd))

(defn set-learning-rate-tracker [builder tracker]
  (.setLearningRateTracker builder tracker))

(defn build [builder]
  (.build builder))
