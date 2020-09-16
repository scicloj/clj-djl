(ns clj-djl.modality.cv
  (:import [ai.djl.modality.cv ImageFactory]))

(defn download-image-from [url]
  (.fromUrl (ImageFactory/getInstance) url))
