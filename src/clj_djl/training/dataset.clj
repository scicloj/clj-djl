(ns clj-djl.training.dataset
  (:import [ai.djl.training.dataset Dataset
            ArrayDataset ArrayDataset$Builder]
           [ai.djl.ndarray NDArray]))

(defn set-sampling [builder batch-size drop-last]
  (.setSampling builder batch-size drop-last)
  builder)

(defn build [builder]
  (.build builder))

(defn prepare
  ([ds]
   (.prepare ds)
   ds)
  ([ds progress]
   (.prepare ds progress)
   ds))

(defn new-array-dataset-builder []
  (ArrayDataset$Builder.))

(defn set-data [builder & data]
  (.setData builder (into-array NDArray data))
  builder)

(defn get-data [ds manager]
  (.getData ds manager))

(defn opt-labels [builder & labels]
  (.optLabels builder (into-array NDArray labels))
  builder)

(defn get-batch-data [batch]
  (.getData batch))

(defn get-batch-labels [batch]
  (.getLabels batch))

(defn close-batch [batch]
  (.close batch))

(defn opt-usage [builder usage]
  (.optUsage builder usage)
  builder)

#_(defn build-dataset [config]
  (let [{:keys [dataset usage sampling]} config]
    ))
