(ns clj-djl.model
  (:refer-clojure :exclude [load])
  (:import [ai.djl Model]))

(defn new-instance [name]
  (Model/newInstance name))

(defn set-block [model block]
  (.setBlock model block)
  model)

(defn set-property [model k v]
  (.setProperty model k v)
  model)

(defn save [model dir name]
  (.save model dir name)
  model)

(defn new-predictor [model translator]
  (.newPredictor model translator))

(defn predict [predictor img]
  (.predict predictor img))

(defn load [model dir]
  (let [model-dir (java.nio.file.Paths/get dir (into-array [""]))]
    (.load model model-dir)
    model))
