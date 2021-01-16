(ns clj-djl.training.loss
  (:import [ai.djl.ndarray NDList]
           [ai.djl.training.loss Loss]))

(defn l1
  ([]
   (Loss/l1Loss))
  ([name]
   (Loss/l1Loss name))
  ([name weight]
   (Loss/l1Loss name weight)))

(def l1-loss l1)

(defn l2
  ([]
   (Loss/l2Loss))
  ([name]
   (Loss/l2Loss name))
  ([name weight]
   (Loss/l2Loss name weight)))


(def l2-loss l2)

(defn hinge
  ([]
   (Loss/hingeLoss))
  ([name]
   (Loss/hingeLoss name))
  ([name margin weight]
   (Loss/hingeLoss name margin weight)))

(def hinge-loss hinge)

(defn sotfmax-cross-entropy
  ([]
   (Loss/softmaxCrossEntropyLoss))
  ([name]
   (Loss/softmaxCrossEntropyLoss name))
  ([name weight class-axis sparse-label from-logit]
   (Loss/softmaxCrossEntropyLoss name weight class-axis sparse-label from-logit)))

(def sotfmax-cross-entropy-loss sotfmax-cross-entropy)

(defn sigmoid-binary-cross-entropy
  ([]
   (Loss/sigmoidBinaryCrossEntropyLoss))
  ([name]
   (Loss/sigmoidBinaryCrossEntropyLoss name))
  ([name weight from-sigmoid]
   (Loss/sigmoidBinaryCrossEntropyLoss name weight from-sigmoid)))

(def sigmoid-binary-cross-entropy-loss sigmoid-binary-cross-entropy)

(defn masked-softmax-cross-entropy
  ([]
   (Loss/maskedSoftmaxCrossEntropyLoss))
  ([name]
   (Loss/maskedSoftmaxCrossEntropyLoss name))
  ([name weight class-axis sparse-label from-logit]
   (Loss/maskedSoftmaxCrossEntropyLoss name weight class-axis sparse-label from-logit)))

(def masked-softmax-cross-entropy-loss masked-softmax-cross-entropy)

(defn evaluate [loss label pred]
  (condp = (vector? label)
    true (.evaluate loss (NDList. label) (NDList. [pred]))
    false (.evaluate loss (NDList. [label]) (NDList. [pred]))))
