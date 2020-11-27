(ns clj-djl.nn
  (:require
   [clj-djl.ndarray :as nd]
   [clj-djl.utils :as utils])
  (:import
   [ai.djl.nn Activation SequentialBlock]
   [ai.djl.nn.core Linear]
   [ai.djl.nn Blocks]
   [ai.djl.training.initializer NormalInitializer]
   [ai.djl.nn.convolutional Conv2d]
   [ai.djl.nn.norm BatchNorm Dropout]
   [ai.djl.ndarray.types Shape]))

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

(defn batchnorm-block [& {:keys [axis center epsilon momentum scale]}]
  (cond-> (BatchNorm/builder)
    axis (.optAxis axis)
    center (.optCenter center)
    epsilon (.optEspilon epsilon)
    momentum (.optMomentum momentum)
    scale (.optScale scale)
    true (.build)))


(defn cov2d-block [{:keys [kernel-shape filters bias dilation groups padding stride]}]
  (cond-> (Conv2d/builder)
    kernel-shape (.setKernelShape (if (sequential? kernel-shape)
                                    (nd/new-shape kernel-shape)
                                    kernel-shape))
    filters (.setFilters filters)
    bias (.optBias bias)
    dilation (.optDilation dilation)
    groups (.optGroups groups)
    padding (.optPadding padding)
    stride (.optStride stride)
    true (.build)))



(defn opt-bias [builder bias]
  (.optBias builder bias))

(defn set-units [builder unit]
  (.setUnits builder unit))

(defn build [builder]
  (.build builder))

(defn add [net block]
  (let [block (if (ifn? block)
                (utils/as-function block)
                block)]
    (.add net block))
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

(defn initialize [block manager datatype & input-shapes]
  (let [datatype (nd/convert-datatype datatype)]
    (.initialize block manager datatype (into-array Shape (map #(nd/new-shape %) input-shapes)))
    block))

(defn get-parameters [block]
  (.getParameters block))

(defn clear [block]
  (.clear block))

(defn dropout [{:keys [rate]}]
  (cond-> (Dropout/builder)
    rate (.optRate rate)
    :always (.build)))
