(ns clj-djl.ndarray
  (:import [ai.djl.ndarray NDArray NDManager]
           [ai.djl.ndarray.types Shape]))

(def manager (NDManager/newBaseManager))

(defn zeros
  ([shape]
   (.zeros manager shape))
  ([m n]
   (zeros (shape m n))))

(defn ones
  ([shape]
   (.ones manager shape))
  ([m n]
   (ones (shape m n))))

(defn shape [m n]
  (Shape. [m n]))

(defn arange [start end]
  (.arange manager start end))
