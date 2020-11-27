(ns clj-djl.utils)

(defmacro try-let [assignments & more]
  `(try
     (let [~@assignments]
       ~@more)
     (catch Exception e#
       (println e#))))

#_(defmacro as-function [f]
  `(reify java.util.function.Function
     (apply [this arg#]
       (~f arg#))))

(defn ^java.util.function.Function as-function [f]
  (reify java.util.function.Function
    (apply [this arg] (f arg))))
