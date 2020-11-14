(ns clj-djl.device
  (:import [ai.djl Device]))

(defn default-device []
  (Device/defaultDevice))

(defn get-devices [maxgpu]
  (Device/getDevices maxgpu))
