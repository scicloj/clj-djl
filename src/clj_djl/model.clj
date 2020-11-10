(ns clj-djl.model
  (:refer-clojure :exclude [load])
  (:import [ai.djl Model]))

(defn new-instance [name]
  (Model/newInstance name))

(defn set-block [model block]
  (.setBlock model block)
  model)

(defn set-datatype [model data-type]
  (.setDataType model data-type)
  model)

(defn get-block [model]
  (.getBlock model))

(defn set-property [model k v]
  (.setProperty model k v)
  model)

(defn new-model [{:keys [name block data-type]}]
  (cond-> (Model/newInstance name)
    block (set-block block)
    data-type (set-datatype data-type)))

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

(defn new-trainer [model config]
  (.newTrainer model config))

(defn get-parameters [layer]
  (.getParameters layer))
