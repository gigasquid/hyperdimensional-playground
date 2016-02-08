(ns hyperdimensional-playground.core
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.linear :as ml]))

(m/set-current-implementation :vectorz)

;;;;;;;;;;

(def sz 10000)

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

a ;=> #vectorz/vector Large vector with shape: [100000]

(m/dot c c)


(defn cosine-sim [v1 v2]
  (/ (m/dot v1 v2)
    (* (ml/norm v1) (ml/norm v2))))

;; because of hyperdimensionality any random vectors will be unrelated

(cosine-sim d d ) ;=>  1.0


(cosine-sim a a) ;=>  1.0
(cosine-sim d d) ;=> 1.0
(cosine-sim a b) ;=>  0.0859992468320239
(cosine-sim b c) ;=> 0.09329186588790261
(cosine-sim a c) ;=> 0.08782018973001954


;;; representing sets with sums

;; you can add the vectors the sum-vector

(defn mean-add [& hvs]
  (m/emap #(Math/round (double %))
   (m/div (apply m/add hvs) (count hvs))))


(defn inverse [hv]
  (m/emap #(- 1 %) hv))

;; x = a + b
(def x (mean-add a b))
(cosine-sim x a) ;=> 0.7234918835988526
(cosine-sim x b) ;=> 0.7229599321393805

;; still not like c at all
(cosine-sim x d)

;; y = x - b = similar to a
(def y (mean-add x (inverse b)))
(cosine-sim y a)
(cosine-sim y b)

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
(hamming-dist xa ya) ;=> 1740.0
(hamming-dist x y) ;=> 1740.0

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
(def x (rand-hv)) ;; name
(def y (rand-hv)) ;; cute-animal
(def z (rand-hv)) ;; favorite-sock
(def a (rand-hv)) ;; Gigasquid
(def b (rand-hv)) ;; duck
(def c (rand-hv)) ;; red-plaid

;H = X * A + Y * B + Z * C
(def h (mean-add (xor-mul x a) (xor-mul y b) (xor-mul z c)))

;; to find the value bound to x (first name) we multiply by the
;; inverse of x and and then match up with the known value  (x is its
;; own xor inverse
(hamming-dist (xor-mul x (xor-mul x a)) a) ;=> 0
(hamming-dist a (xor-mul x h)) ;=> 1462.0
(hamming-dist b (xor-mul x h)) ;=> 1721.0
(hamming-dist c (xor-mul x h)) ;=> 1736.0
(cosine-sim a (xor-mul x h)) ;=> 0.3195059768353112
(cosine-sim b (xor-mul x h)) ;=> 0.1989075567830733
(cosine-sim c (xor-mul x h)) ;=> 0.18705233578983288

(hamming-dist (xor-mul x (xor-mul x a)) a) ;=> 0
(hamming-dist a (xor-mul y h)) ;=> 1462.0
(hamming-dist b (xor-mul y h)) ;=> 1721.0
(hamming-dist c (xor-mul y h)) ;=> 1736.0
(cosine-sim a (xor-mul y h)) ;=> 0.3195059768353112
(cosine-sim b (xor-mul y h)) ;=> 0.1989075567830733
(cosine-sim c (xor-mul y h)) ;=> 0.18705233578983288
