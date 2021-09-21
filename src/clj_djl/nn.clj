(ns clj-djl.nn
  (:require
   [clj-djl.ndarray :as nd]
   [clj-djl.model :as m]
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

(defn leaky-relu-block
  "Create a LamdaBlock with LeakyReLU as forward function:

  LeakyRelu = (if (>= x 0) x (* neg_slope x))"
  [alpha]
  (Activation/leakyReluBlock (float alpha)))

(defn leaky-relu
  [data alpha]
  (Activation/leakyRelu data (float alpha)))

(defn prelu-block
  "Creates a LambdaBlock that applies the PreLU activation function in its forward
  function, the neg_slope is learnt during training"
  []
  (Activation/preluBlock))

(defn elu-block
  "Creates a LambdaBlock that applies the ELU activation function in its forward
  function

  ELU <- (if (> x 0) x (* alpha (- (pow e x) 1)))"
  [alpha]
  (Activation/eluBlock (float alpha)))

(defn elu
  "Applies ELU(Exponential Linear Unit) activation on the input NDArray or
  NDList"
  [data alpha]
  (Activation/elu data (float alpha)))

(defn gelu-block
  "Creates a LambdaBlock that applies the GELU activation function in its forward
  function"
  []
  (Activation/geluBlock))

(defn gelu
  "Applies GELU(Gausian Error Linear Unit) activation on the input NDArray or
  NDList"
  [data]
  (Activation/gelu data))

(defn selu-block
  "Creates a LambdaBlock that applies the SELU activation function in its forward
  function

  SELU <- (* lambda (if (> x 0) x (* alpha (- (pow e x) 1)))), where lamda is
  1.0507009873554804934193349852946 and alpha is
  1.6732632423543772848170429916717"
  []
  (Activation/seluBlock))

(defn selu
  "Applies SELU(Scaled Exponential Linear Unit) activation on the input NDArray or
  NDList"
  [data]
  (Activation/selu data))

(defn swish-block
  "Creates a LambdaBlock that applies the Swish activation function in its forward
  function"
  [beta]
  (Activation/swishBlock (float beta)))

(defn swish
  "Applies Swish activation on the input NDArray or NDList"
  [data beta]
  (Activation/swish data (float beta)))

(defn mish-block
  "Creates a LambdaBlock that applies the Mish activation function in its forward
  function"
  []
  (Activation/mishBlock))

(defn mish
  "Applies Mish activation on the input NDArray or NDList"
  [data]
  (Activation/mish data))

(defn linear-builder []
  (Linear/builder))

(def new-linear-builder linear-builder)

(defn linear [{:keys [bias units]}]
  (cond-> (Linear/builder)
    bias (.optBias bias)
    units (.setUnits units)
    true (.build)))

(def linear-block linear)

(defn batchnorm-block
  ([]
   (.build (BatchNorm/builder))    )
  ([{:keys [axis center epsilon momentum scale]}]
   (cond-> (BatchNorm/builder)
     axis (.optAxis axis)
     center (.optCenter center)
     epsilon (.optEspilon epsilon)
     momentum (.optMomentum momentum)
     scale (.optScale scale)
     true (.build))))


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

(defn set-initializer [net initializer parameter]
  (.setInitializer net initializer parameter)
  net)

(defn normal-initializer
  ([]
   (NormalInitializer.))
  ([sigma]
   (NormalInitializer. sigma)))

(def new-normal-initializer normal-initializer)

(defn initialize [block manager datatype- & input-shapes]
  (let [datatype (nd/datatype datatype-)]
    (.initialize block manager datatype (into-array Shape (map #(nd/shape %) input-shapes)))
    block))

(defn forward
  ([block inputs]
   (let [ndm (.getManager inputs)
         _ (initialize block ndm :float32 (nd/shape inputs))
         model (m/model {:name "lin-reg" :block block})
         translator (ai.djl.translate.NoopTranslator. nil)
         predictor (.newPredictor model translator)]
     (nd/get (.predict predictor (nd/ndlist inputs)) 0)))
  ([block paramstore inputs labels-or-training? & [params]]
   (if (nil? params)
     (.forward block paramstore inputs labels-or-training?)
     (.forward block paramstore inputs labels-or-training? params))))

(defn get-parameters [block]
  (.getParameters block))

(defn clear [block]
  (.clear block))

(defn dropout [{:keys [rate]}]
  (cond-> (Dropout/builder)
    rate (.optRate rate)
    :always (.build)))

(defn sequential
  ([]
   (SequentialBlock.))
  ([{:keys [blocks initializer parameter]}]
   (cond-> (SequentialBlock.)
     blocks (.addAll (into-array ai.djl.nn.Block (if (instance? ai.djl.nn.Block blocks)
                                                   [blocks]
                                                   blocks)))
     initializer (set-initializer initializer parameter))))

(def sequential-block sequential)
