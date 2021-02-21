(ns clj-djl.model
  (:refer-clojure :exclude [load])
  (:import [ai.djl Model]))

(defn instance [name]
  (Model/newInstance name))

(def new-instance instance)

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

(defn model [{:keys [name block data-type]}]
  (cond-> (Model/newInstance name)
    block (set-block block)
    data-type (set-datatype data-type)))

(def new-model model)

(defn save [model dir name]
  (if (string? dir)
    (.save model (java.nio.file.Paths/get dir (into-array [""])) name)
    (.save model dir name))
  model)

(defn predictor [model translator]
  (.newPredictor model translator))

(def new-predictor predictor)

(defn predict [predictor img]
  (.predict predictor img))

(defn load
  ([model dir]
   (let [model-dir (java.nio.file.Paths/get dir (into-array [""]))]
     (.load model model-dir)
     model))
  ([model dir name]
   (if (string? dir)
     (.load model (java.nio.file.Paths/get dir (into-array [""])) name)
     (.load model dir name))
   model))

(defn trainer [model config]
  (.newTrainer model config))

(def new-trainer trainer)

(defn get-parameters [layer]
  (.getParameters layer))

(defn get-ndmanager [model]
  (.getNDManager model))

(defn clear [block]
  (.clear block))
