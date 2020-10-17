(ns clj-djl.device
  (:import [ai.djl Device]))

(defn default-device []
  (Device/defaultDevice))
