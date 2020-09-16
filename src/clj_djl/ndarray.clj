(ns clj-djl.ndarray
  (:import [ai.djl.ndarray NDArray NDManager]
           [ai.djl.ndarray.types Shape]))

(def manager (NDManager/newBaseManager))

(defn shape [m n]
  (Shape. [m n]))

(defn get-shape [ndarray]
  (.getShape ndarray))

(defn reshape [ndarray new-shape]
  (.reshape ndarray (long-array new-shape)))

(defn size [ndarray]
  (.size ndarray))

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

(defn arange [start end]
  (.arange manager start end))
