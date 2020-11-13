(ns clj-djl.nn
  (:import [ai.djl.nn Activation SequentialBlock]
           [ai.djl.nn.core Linear]
           [ai.djl.nn Blocks]
           [ai.djl.training.initializer NormalInitializer]))

(defn relu-block []
  (Activation/reluBlock))

(defn relu [data]
  (Activation/relu data))

(defn sigmoid-block []
  (Activation/sigmoidBlock))

(defn sigmoid [data]
  (Activation/sigmoid data))

(defn tanh-block []
  (Activation/tanhBlock))

(defn tanh [data]
  (Activation/tanh data))

(defn softplus-block []
  (Activation/softPlusBlock))

(defn softplus
  [data]
  (Activation/softPlus data))

(defn sequential-block []
  (SequentialBlock.))

(defn new-linear-builder []
  (Linear/builder))

(defn linear-block [{:keys [bias units]}]
  (cond-> (Linear/builder)
    bias (.optBias bias)
    units (.setUnits units)
    true (.build)))

(defn opt-bias [builder bias]
  (.optBias builder bias))

(defn set-units [builder unit]
  (.setUnits builder unit))

(defn build [builder]
  (.build builder))

;; defmulti
(defn add [net block]
  (.add net block)
  net)

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

(defn set-initializer [net initializer]
  (.setInitializer net initializer)
  net)

(defn new-normal-initializer
  ([]
   (NormalInitializer.))
  ([sigma]
   (NormalInitializer. sigma)))
