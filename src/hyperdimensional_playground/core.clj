(ns hyperdimensional-playground.core
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.linear :as ml]))

(m/set-current-implementation :vectorz)

;;;;;;;;;;

(def sz 10000)

;; an entity is given a random hypervector

(defn rand-hv []
  (let [hv (m/new-sparse-array [sz])]
    (dotimes [i (rand-int sz)]
      (m/mset! hv (rand-int sz) 1))
    hv))

(def a (rand-hv))
(def b (rand-hv))
(def c (rand-hv))
(def d (rand-hv))

(m/dot c c)

(defn cosine-sim [v1 v2]
  (/ (m/dot v1 v2)
     (* (ml/norm v1) (ml/norm v2))))
;; because of hyperdimensionality any random vectors will be unrelated

(cosine-sim c c) ;=> 1.0
(cosine-sim d d) ;=> 1.0
(cosine-sim d c) ;=> 0.18960582948561497
(cosine-sim d b) ;=> 0.15069209159422153
(cosine-sim c b) ;=> 0.08945823871209621


;;; representing sets with sums

;; you can add the vectors the sum-vector

(defn mean-add [& hvs]
  (m/emap #(Math/round %)
   (m/div (apply m/add hvs) (count hvs))))

(def e (m/add b c))
(def e (mean-add b c))

(cosine-sim c e) ;=> 0.7842166543753146
(cosine-sim b e) ;=> 0.6881539417482678
(cosine-sim a e) ;=> 0.23200400071556893

(def f (m/sub e b)) ;; should only be c now
(cosine-sim f c) ;=> 1.0

(cosine-sim (m/mul a b) (m/mul b a)) ;=> 1

(defn xor-mul [v1 v2]
  (->> (m/add v1 v2)
      (m/emap #(mod % 2))))

(defn hamming-dist [v1 v2]
  (m/esum (xor-mul v1 v2)))

(hamming-dist [1 0 1] [1 0 1]) ;=> 0
(hamming-dist [1 0 1 1 1 0 1] [1 0 0 1 0 0 1]) ;=> 2
(hamming-dist a a) ;=> 0

;; multiplication randomizes but preserves the distance
(def x (rand-hv))
(def y (rand-hv))
(def xa (xor-mul x a))
(def ya (xor-mul y a))
(hamming-dist xa ya) ;=> 4938.0
(hamming-dist x y) => 4938.0

(def z (rand-hv))

;; multiplication distributes over addition
(hamming-dist (xor-mul a (m/add x y z))
              (m/add (xor-mul x a)
                     (xor-mul x y)
                     (xor-mul x z))) ;=> 0


;; permutations randomize
;; representing sequences by permutating sums
(def d (rand-hv))
(def e (rand-hv))

(def s1 ( (m/rotate d 0 1) e))
;; we can probe memory for d in sequence, not by d but by rotated d.
;; you can then get e by subtracting rotated d and and probing mem for
;; result

(def c (rand-hv))

(defn gen-hv-seq [hvecs]
  (let [f (first hvecs)
        r (rest hvecs)]
    (if (seq r)
      (mean-add (m/rotate (gen-hv-seq r) 0 1) f)
      f)))

(def s (gen-hv-seq [c d e]))

;; data records with bound pairs
(def x (rand-hv)) ;first name
(def y (rand-hv)) ;last name
(def z (rand-hv)) ;age
(def a (rand-hv)) ;Giga
(def b (rand-hv)) ;Squid
(def c (rand-hv)) ;101
;H = X * A + Y * B + Z * C
(def h (mean-add (xor-mul x a) (xor-mul y b) (xor-mul z c)))

;; to find the value bound to x (first name) we multiply by the
;; inverse of x and and then match up with the known value  (x is its
;; own xor inverse
(hamming-dist (xor-mul x (xor-mul x a)) a)
(hamming-dist a (xor-mul x h)) ;=>  2700.0  closest to Giga known
(hamming-dist b (xor-mul x h)) ;=> 5289.0
(hamming-dist c (xor-mul x h)) ;=> 4901.0
(cosine-sim a (xor-mul x h)) ;=> 0.707183394302094
(cosine-sim b (xor-mul x h)) ;=> 0.2930065638109825
(cosine-sim c (xor-mul x h)) ;=> 0.17250353971137838
