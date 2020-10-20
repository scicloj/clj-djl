(ns clj-djl.training.tracker
  (:import [ai.djl.training.tracker Tracker]))

(defn fixed [value]
  (Tracker/fixed value))
