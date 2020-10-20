(ns clj-djl.training.listener
  (:import [ai.djl.training.listener TrainingListener$Defaults]))

(defn logging []
  (TrainingListener$Defaults/logging))
