(ns clj-djl.device
  (:import [ai.djl Device]))

(defn cpu []
  (Device/cpu))

(defn gpu []
  (Device/gpu))
