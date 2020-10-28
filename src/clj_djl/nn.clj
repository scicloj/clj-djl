(ns clj-djl.nn
  (:import [ai.djl.nn Activation SequentialBlock]
           [ai.djl.nn.core Linear]
           [ai.djl.nn Blocks]))

(defn relu-block []
  (Activation/reluBlock))

(defn relu [data]
  (Activation/relu data))

(defn sigmoid-block []
  (Activation/sigmoidBlock))

(defn sigmoid [data]
  (Activation/sigmoid data))

(defn sequential-block []
  (SequentialBlock.))

(defn new-linear-builder []
  (Linear/builder))

(defn opt-bias [builder bias]
  (.optBias builder bias))

(defn set-units [builder unit]
  (.setUnits builder unit))

(defn build [builder]
  (.build builder))

(defn add [net block]
  (.add net block))

(defn batch-flatten-block [& more]
  (if (nil? more)
    (Blocks/batchFlattenBlock)
    (Blocks/batchFlattenBlock (first more))))

(defn batch-flatten [array & more]
  (if (nil? more)
    (Blocks/batchFlatten array)
    (Blocks/batchFlatten array (first more))))

(defn identity-block []
  (Blocks/identityBlock))

(defn forward [block paramstore inputs labels-or-training? & [params]]
  (if (nil? params)
    (.forward block paramstore inputs labels-or-training?)
    (.forward block paramstore inputs labels-or-training? params)))
