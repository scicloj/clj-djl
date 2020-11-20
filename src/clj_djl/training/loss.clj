(ns clj-djl.training.loss
  (:import [ai.djl.ndarray NDList]
           [ai.djl.training.loss Loss]))

(defn l1-loss
  ([]
   (Loss/l1Loss))
  ([name]
   (Loss/l1Loss name))
  ([name weight]
   (Loss/l1Loss name weight)))

(defn l2-loss
  ([]
   (Loss/l2Loss))
  ([name]
   (Loss/l2Loss name))
  ([name weight]
   (Loss/l2Loss name weight)))

(def l2 l2-loss)

(defn hinge-loss
  ([]
   (Loss/hingeLoss))
  ([name]
   (Loss/hingeLoss name))
  ([name margin weight]
   (Loss/hingeLoss name margin weight)))

(defn sotfmax-cross-entropy-loss
  ([]
   (Loss/softmaxCrossEntropyLoss))
  ([name]
   (Loss/softmaxCrossEntropyLoss name))
  ([name weight class-axis sparse-label from-logit]
   (Loss/softmaxCrossEntropyLoss name weight class-axis sparse-label from-logit)))

(defn sigmoid-binary-cross-entropy-loss
  ([]
   (Loss/sigmoidBinaryCrossEntropyLoss))
  ([name]
   (Loss/sigmoidBinaryCrossEntropyLoss name))
  ([name weight from-sigmoid]
   (Loss/sigmoidBinaryCrossEntropyLoss name weight from-sigmoid)))

(defn masked-softmax-cross-entropy-loss
  ([]
   (Loss/maskedSoftmaxCrossEntropyLoss))
  ([name]
   (Loss/maskedSoftmaxCrossEntropyLoss name))
  ([name weight class-axis sparse-label from-logit]
   (Loss/maskedSoftmaxCrossEntropyLoss name weight class-axis sparse-label from-logit)))

(defn evaluate [loss label pred]
  (condp = (vector? label)
    true (.evaluate loss (NDList. label) (NDList. [pred]))
    false (.evaluate loss (NDList. [label]) (NDList. [pred]))))
