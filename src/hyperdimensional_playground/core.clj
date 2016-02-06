(ns hyperdimensional-playground.core
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.linear :as ml]))

(m/set-current-implementation :vectorz)

;;;;;;;;;;

(def sz 100000)
(Math/round (double (* 10 (/ 1 10))))

;; an entity is given a random hypervector

;; assume that 100,000 is big enough to be sparse with less than 10% of
;; it having 1s
(defn rand-hv []
  (let [hv (m/new-sparse-array [sz])
        n (* 0.1 sz)]
    (dotimes [i n]
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
(cosine-sim d c) ;=> 0.0
(cosine-sim a b) ;=> 0.0
(cosine-sim b c) ;=> 0.0
(cosine-sim a c) ;=> 0.0


;;; representing sets with sums

;; you can add the vectors the sum-vector

(defn mean-add [& hvs]
  (m/emap #(Math/round (double %))
   (m/div (apply m/add hvs) (count hvs))))


(defn inverse [hv]
  (m/emap #(- 1 %) hv))

;; x = a + b
(def x (mean-add a b))
(cosine-sim x a) ;=> 0.858395075278952
(cosine-sim x b) ;=> 0.512989176042577
(cosine-sim x c) ;=> 0.0

;; y = x - b = similar to a
(def y (mean-add x (inverse b)))
(cosine-sim y a) ;=>  0.07483314773547883  (most similar)
(cosine-sim y b) ;=>  0.044721359549995794

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
(hamming-dist xa ya) ;=>  80.0
(hamming-dist x y) ;=> 80.0

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

(def s1 (xor-mul (m/rotate d 0 1) e))

;; we can probe memory for d in sequence, not by d but by rotated d.
;; you can then get e by subtracting rotated d and and probing mem for
;; result
(cosine-sim s1 (m/rotate d 0 1)) ;=> 0.6697388721247042
(def r (mean-add (inverse (m/rotate d 0 1))))
(cosine-sim s1 e) ;=-> 0.667150039013408


(defn gen-hv-seq [hvecs]
  (let [f (first hvecs)
        r (rest hvecs)]
    (if (seq r)
      (mean-add (m/rotate (gen-hv-seq r) 0 1) f)
      f)))


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
(hamming-dist (xor-mul x (xor-mul x a)) a) ;=> 0
(hamming-dist a (xor-mul x h));=> 83.0 Closest to Giga
(hamming-dist b (xor-mul x h)) ;=> 62.0
(hamming-dist c (xor-mul x h)) ;=> 81.0
(cosine-sim a (xor-mul x h)) 0.0
(cosine-sim b (xor-mul x h))
(cosine-sim c (xor-mul x h))
