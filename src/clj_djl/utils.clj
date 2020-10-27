(ns clj-djl.utils)

(defmacro try-let [assignments & more]
  `(try
     (let [~@assignments]
       ~@more)
     (catch Exception e#
       (println e#))))
