(ns hyperdimensional-playground.core
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.linear :as ml]))

(m/set-current-implementation :vectorz)

;;; Size of the hypervectors
(def sz 10000)

(defn rand-hv
"Generate a random hypervector"
  []
  (let [hv (m/new-sparse-array [sz])
        n (* 0.1 sz)]
    (dotimes [i n]
      (m/mset! hv (rand-int sz) 1))
    hv))

(defn cosine-sim
  "Cosine similarity between two hypervectors"
  [v1 v2]
  (let [norm1 (ml/norm v1)
        norm2 (ml/norm v2)]
    (when (and (pos? norm1) (pos? norm2))
      (/ (m/dot v1 v2) (* norm1 norm2)))))

(defn mean-add
  "Add two binary hypervectors together with mean rounding"
  [& hvs]
  (m/emap #(Math/round (double %))
   (m/div (apply m/add hvs) (count hvs))))

(defn inverse
  "Inverse of a hypervector for subtracting"
 [hv]
  (m/emap #(- 1 %) hv))

(defn xor-mul
  "XOR multiplication for binary hypervectors"
  [v1 v2]
  (->> (m/add v1 v2)
      (m/emap #(mod % 2))))

(defn hamming-dist
  "Hamming distance or number of bits different in 2 vectors"
  [v1 v2]
  (m/esum (xor-mul v1 v2)))

(defn gen-hv-seq
  "Generate a sequence of hypervectors using rotation"
  [hvecs]
  (let [f (first hvecs)
        r (rest hvecs)]
    (if (seq r)
      (mean-add (m/rotate (gen-hv-seq r) 0 1) f)
      f)))
