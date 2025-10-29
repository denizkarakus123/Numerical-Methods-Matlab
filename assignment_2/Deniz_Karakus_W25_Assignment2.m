%% ECSE 343: Numerical Methods for Engineers- Assignment 2
% 
% 
% Due Date: 22nd Feb. 2024
% 
% Student Name: Deniz Karakus
% 
% Student ID: 261114510
%% Please type your answers and write you code in this .mlx script. If you choose to provide the handwritten answers, please scan your answers and include those in SINGLE pdf file.
%% Please submit this *.mlx* file along with the *PDF* copy of this file.
%% 
% 
%% ECSE 343: Numerical Methods for Engineers- Assignment 1- part b
% 
% 
% Due Date: 25 Feb. 2025
% 
% Student Name: Deniz Karakus
% 
% Student ID: 261114510
%% Please type your answers and write you code in this .mlx script. If you choose to provide the handwritten answers, please scan your answers and include those in SINGLE pdf file.
%% Please submit this *.mlx* file along with the *PDF* copy of this file.
% *Table of Contents*
% 
% Please type your answers and write you code in this .mlx script. If you choose 
% to provide the handwritten answers, please scan your answers and include those 
% in SINGLE pdf file.
Please submit this .mlx file along with the PDF copy of 
% this file.
Question 1: Cholesky Least Squares Estimation (10 marks)
Question 
% 2: Proofs (10 marks)
Question 3:  QR Decomposition (10 marks)
Question 4:  Polynomial 
% Fitting using Least Squares Estimation  (10 marks)
Appendix 
 Implement the 
% Cholesky Decompositon fucntion here.
Implement Least Square Solver for Cholesky 
% here
Implement Polynomial Matrix here 
Implement QR Methods here
Function to 
% generate Hilbert Matrix is provided here
%% 
% 
%% Question 1: Proofs *(10 marks)*
% a) Show that the condition number of $\textbf A^T \textbf A$ is $\kappa \;\left({\mathit{\mathbf{A}}}^{T\;} 
% \mathit{\mathbf{A}}\right)={\kappa \left(\mathit{\mathbf{A}}\right)}^2 \;$.
% 
% *First, recall that the singular values of A, denoted sigma_i(A), are defined 
% as the square roots of the eigenvalues of A^T A. In symbols, sigma_i(A) = sqrt(lambda_i(A^T 
% A)), where lambda_i(A^T A) are the eigenvalues of A^T A. The condition number 
% of A is then k(A) = sigma_max(A) / sigma_min(A).*
% 
% *Next, consider A^T A itself. Since A^T A is symmetric (or Hermitian) and 
% positive semidefinite, its singular values coincide with its eigenvalues. That 
% is, for A^T A we have sigma_i(A^T A) = lambda_i(A^T A). By definition, the condition 
% number of A^T A is k(A^T A) = sigma_max(A^T A) / sigma_min(A^T A). But sigma_max(A^T 
% A) = lambda_max(A^T A) and sigma_min(A^T A) = lambda_min(A^T A).*
% 
% *We also know that lambda_max(A^T A) = [sigma_max(A)]^2 and lambda_min(A^T 
% A) = [sigma_min(A)]^2, because the eigenvalues of A^T A are the squares of the 
% singular values of A. Substituting these into the condition number of A^T A 
% gives:*
% 
% *k(A^T A) = [sigma_max(A)]^2 / [sigma_min(A)]^2 = (sigma_max(A) / sigma_min(A))^2 
% = [k(A)]^2.*
% 
% *Therefore, we conclude that k(A^T A) = [k(A)]^2.*
% 
% 
%% 
% 
% 
% b) Given two orthonormal matrices $U$and $Q$, show that product of these matrices, 
% $\textrm{UQ}$, is also orthonormal.
% 
% *Let U and Q be two orthonormal matrices of compatible dimensions. By definition 
% of normality, we have U^T U = I, and Q^T Q = I. We want to show that (UQ) is 
% also orthonormal. We want to compute (UQ)^T (UQ) and show that its equal to 
% I.  (UQ)^T (UQ) = Q^T U^T UQ. Since U^T U = I, This simplifies to (UQ)^T (UQ) 
% = Q^T IQ = Q^T Q = I. Therefore, (UQ)^T (UQ) = I. This means that U and Q are 
% orthonormal.* 
% 
% 
%% 
% c) If A is an invertible matrix and *Q* is an orthonormal matrix, show that 
% $\kappa \left(\textrm{QA}\right)=\kappa \left(A\right)\ldotp$
% 
% *Let Q be an n×n orthonormal matrix, so Q^T Q = I, and let A be an n×n invertible 
% matrix. We want to show that multiplying A on the left by Q does not change 
% its condition number.*
%% 
% # *Consider the product QA. Compute (QA)^T (QA): (QA)^T (QA) = A^T Q^T Q A. 
% Since Q^T Q = I, this simplifies to (QA)^T (QA) = A^T A.*
% # *The singular values of any matrix M are the square roots of the eigenvalues 
% of M^T M. Therefore, the eigenvalues of (QA)^T (QA) are the same as the eigenvalues 
% of A^T A. Hence, the singular values of QA match those of A.*
% # *The 2-norm condition number of a matrix M is defined as k(M) = sigma_max(M) 
% / sigma_min(M), where sigma_max(M) and sigma_min(M) are the largest and smallest 
% singular values of M, respectively. Since QA and A share the same singular values, 
% we have k(QA) = sigma_max(QA) / sigma_min(QA) = sigma_max(A) / sigma_min(A) 
% = k(A).*
%% 
% *Thus, multiplying A by an orthonormal matrix Q does not affect its condition 
% number, so k(QA) = k(A).*
%% 
%% 
%% 
% d) Given a full rank matrix $\textbf  A \in \mathbb R^{n \times n}$, prove 
% that $\textbf A^T \textbf A$ is symetteric and positive definite.
% 
% *Symmetry:*
% 
% *Compute the transpose of A^T A. We get (A^T A)^T = A^T (A^T)^T = A^T A. Therefore, 
% A^T A is symmetric.* 
% 
% *Positive definite:* 
% 
% *A matrix M is positive definite if, for every nonzero vector x, we have x^T 
% M x > 0. Let x be any nonzero vector in R^n. Then*
% 
% *x^T (A^T A)x = (Ax)^T (Ax) = || Ax ||^2.*
% 
% *Since A is invertible, A x = 0 if and only if x = 0. Thus, for x ≠ 0, we 
% must have A x ≠ 0. Hence* 
% 
% *||Ax||^2 > 0. Therefore,*
% 
% *x^T (A^T A) x > 0 for all x ≠ 0,*
% 
% *which shows A^T A is positive definite.*
%% Question 2: Forward and Backward Substitution. (10 marks)
% a) Write the  functions named _forward_sub_ to perform the forward substitution 
% on a lower triangular matrix. Add your function in the appendix of this file.
% 
% You can check your function by comparing the obtained results with the MATLAB's 
% built in backslash operator. 
% 
% Use the cell below to test the correctness of your function on the provided 
% lower triangular matrix.  

 A = tril(magic(5)) % A is a 5x5 lower triangular matrix   
 b = [1; 2; 3; 4;5];     % right hand side vector of size 4x1

%---------------------------------------------------------------------------------------------------------
 % call your forward_sub function and solve  Ax = b    % (where A is a
 % lower triangular square matrix)

 my_solution = forward_sub(A,b);
 matlab_solution = A\b;

 disp('Solution using forward_sub:');
 disp(my_solution);
 disp('Solution using MATLAB backslash operator:');
 disp(matlab_solution);

%% 
% b) Write the _backward_sub_ function to perform the backward substitution 
% on a lower triangular matrix. Add your function in the appendix of this file.
% 
% Use the cell below to test the correctness of your function by comparing your 
% solution with the MATLAB's built in backslash operator.

 A = triu(magic(5)) % A is a 5x5 upper triangular matrix   
 b = [1; 2; 3; 4;5];     % right hand side vector of size 4x1

%---------------------------------------------------------------------------------------------------------
 % call your backward_sub function and solve  Ax = b    % (where A is a
 % upper triangular square matrix)

 my_solution = backward_sub(A,b);
 matlab_solution = A\b;

 disp('Solution using backward_sub:');
 disp(my_solution);
 disp('Solution using MATLAB backslash operator:');
 disp(matlab_solution);

%% Question 3: Solving linear systems of equations. (10 marks)
% a) Write a function named _myLU.m_  in the Appendix of this file to compute 
% the LU factorization using Gaussian Elimination. 
% 
% 
% 
% 
% 
% b)  In the cell below use the LU decomposition implemented in part (c) along 
% with forward and backward substitution functions written in Question 2 part 
% (a) and (b), respectively, to solve the following system of equations. List 
% the values of vector X obtained.  
% 
% $$\underset{A}{\underbrace{{\left[\begin{array}{cccc}1e-16 & 50 & 5 & 9\\0.2 
% & 5 & 7.4 & 5\\0.5 & 4 & 8.5 & 32\\0.89 & 8 & 11 & 92\end{array}\right]}} } 
% \underset{X}{\underbrace{{\left[\begin{array}{c}x_1 \\x_2 \\x_3 \\x_4 \end{array}\right]}} 
% } =\ \underset{b}{\underbrace{{\left[\begin{array}{c}40\\52\\18\\95\end{array}\right]}} 
% }$$

% test your code here
%set up matrix A and vector b
A = [ 1e-16, 50, 5, 9;
      0.2, 5, 7.4, 5;
      0.5, 4, 8.5, 32;
      0.89, 8, 11, 92 ];
b = [40; 52; 18; 95];

%LU decomp
[L, U] = myLU(A);

%solve L*y = b via forward substitution
y = forward_sub(L, b);

%solve U*x = y via backward substitution
x_no_pivot = backward_sub(U, y);

%display the solution
disp('Solution vector x is:');
disp(x_no_pivot);

%compare with MATLAB's built-in solver
matlab_x = A \ b;
disp('MATLAB built-in solution (A\\b):');
disp(matlab_x);
%% 
% c) Write a function named _myPLU.m_ to compute the Gaussian Elimination based 
% LU factorization *using the partial pivoting* (row pivoting). Choose the elements 
% with maximum magnitude as pivots. The function should take matrix  as the input 
% and should output L, U and P matrices. Write your function in the Appendix of 
% this file.
% 
% 
% 
% d)  In the cell below use the LU decomposition with partial pivoting implemented 
% in part (b) along with forward and backward substitution functions written in 
% part (c) to solve the above system of equations. List the values of vector *x* 
% obtained.  

% test your code here 
A = [ 1e-16, 50, 5, 9;
      0.2, 5, 7.4, 5;
      0.5, 4, 8.5, 32;
      0.89, 8, 11, 92 ];
b = [40; 52; 18; 95];
[L, U, P] = myPLU(A);

%apply the permutation to the right-hand side vector
Pb = P * b;

%solve the lower triangular system L*y = P*b using forward substitution
y = forward_sub(L, Pb);

%solve the upper triangular system U*x = y using backward substitution
x_pivot = backward_sub(U, y);

%display the solution vector x obtained from the LU with partial pivoting method
disp('Solution vector x using LU with partial pivoting is:');
disp(x_pivot);

%compare with MATLAB's built-in solver
x_matlab = A \ b;
disp('MATLAB built-in solution (A\\b):');
disp(x_matlab);


%% 
% g) Comment on solutions obtained in part (d) and (f). Use the residual to 
% determine which solution is more accurarte.
% 
% *as we can see when we run the code below, the residual norm is much lower 
% when we use partial pivoting. Therefore, the solution with partial pivoting 
% is much more accurate. This is due to the 1e-16 term in the top-left position 
% of the matrix A, which creates significant numerical instability when used as 
% a pivot without pivoting. When this very small value (essentially zero in floating-point 
% arithmetic) is used as a divisor in the standard LU decomposition, it leads 
% to extremely large multipliers in the L matrix (around 10^16 magnitude). These 
% large multipliers cause severe error amplification during forward and backward 
% substitution.*


r_no_pivot = A*x_no_pivot - b;
r_pivot = A*x_pivot - b;


norm_no_pivot = norm(r_no_pivot);
norm_pivot = norm(r_pivot);

disp('Residual norm without partial pivoting:');
disp(norm_no_pivot);

disp('Residual norm with partial pivoting:');
disp(norm_pivot);
%% Question 4: Using LU and Forward/ Backward Substitution. *(10 marks)*
% You are tasked with designing a computer program for solving a linear system 
% *Ax = b,* where $\textbf  A \in \mathbb R^{n \times n}$ (i.e. a $n\times n$ 
% matrix) , and $\textbf  x, b \in \mathbb R^{n }$. You can assume that the system 
% is very large and is invertible. You have access to the following numerical 
% library functions that you can use in your pseudo-code:
%% 
% # Fundamental matrix and vector operations (multiplication, addition etc.) 
% # PLU(*M*)  Returns L, U and P such that $\mathit{\mathbf{P}}*\mathit{\mathbf{M}}=\mathit{\mathbf{P}}*\mathit{\mathbf{L}}\;\mathit{\mathbf{U}}$ 
% (Partial pivoting). Uses  about $O\left(n^3 \right)$ floating point operations
% # LU(*M*)  Returns L and U such that $\mathit{\mathbf{L}}\;\mathit{\mathbf{U}}=\mathbf{M}$ 
% (No pivoting). Uses  about $O\left(n^3 \right)$ floating point operations.
% # transp(*M*) Returns the transpose of *M* (*M* could be a matrix  or a vector) 
% # Chol(*M*)  Returns L such that $\mathit{\mathbf{M}}={\mathit{\mathbf{U}}}^{\mathit{\mathbf{T}}\;} 
% \mathit{\mathbf{U}}$ (Uses Cholesky decomposition – A must be symmetric positive 
% definite).Uses  about $O\left(\frac{n^3 }{2}\right)$ floating point operations.
% # FwdSub(L,b)  Returns y such that $\mathit{\mathbf{L}}\;\mathit{\mathbf{y}}=\mathit{\mathbf{b}}$ 
% (L must be lower triangular). Uses about $O\left(n^2 \right)$ floating point 
% operations.
% # BwdSub(U,y)  Returns x such that $\mathit{\mathbf{U}}\;\mathit{\mathbf{x}}=\mathit{\mathbf{y}}$ 
% (U must be upper triangular). Uses about $O\left(n^2 \right)$ floating point 
% operations.
%% 
% For each part of this question, justify the overall approach and the specific 
% algorithms used given the requirements and problem description. You need to 
% write an algirithm that
% 
% a)  You need to solve the system ${\mathit{\mathbf{A}}\;\mathit{\mathbf{x}}}_{i\;} 
% ={\mathit{\mathbf{b}}}_{\mathit{\mathbf{i}}}$*,* $i=1\ldotp \ldotp \ldotp s$, 
% where $s<<n$. The right hand side vector changes f for different values of $i$, 
% you need to solve for the unknown vector ${\mathit{\mathbf{x}}}_{\mathit{\mathbf{i}}\;}$. 
% Provide a pseudo-code of the algorithm such that the CPU cost (the number of 
% floating points operations) is minimum.
% 
% Comment on the cost of your algorithm in terms of the number of Forward/Backward 
% substritutions, and LU decomopositions.
% 
% *Algorithm Pseudo-code:* 
% 
% *function SolveMultipleSystems(A, B)*
% 
% *//compute LU decomposition once*
% 
% *[L, U, P] = PLU(A)*
% 
% *//initialize solution matrix*
% 
% *X = zeros(n, s)*
% 
% 
% 
% *//solve for each right-hand side*
% 
% *for i = 1 to s do*
% 
% *//extract current right-hand side*
% 
% *b = B[:,i]*
% 
% *//apply permutation*
% 
% *b_perm = P * b*
% 
% *//forward substitution*
% 
% *y = FwdSub(L, b_perm)*
% 
% 
% 
% *//backward substitution*
% 
% *X[:,i] = BwdSub(U, y)*
% 
% *end for*
% 
% 
% 
% *return X*
% 
% *end function*
% 
% 
% 
% *To solve multiple systems A * x_i = b_i efficiently when s << n, we use LU 
% decomposition with partial pivoting to factorize A once and solve each system 
% using forward and backward substitution.*
% 
% *Total Cost:*
%% 
% * *Overall Complexity: O(n^3) + s * O(n^2)*
% * *This approach is better than solving from scratch for each b_i, which would 
% take O(s * n^3).*
% * *It avoids direct matrix inversion and improves numerical stability.*
%% 
% *By reusing L and U, we reduce computations and ensure efficiency.*
% 
% 
% 
% b)  You need to solve the system ${\mathit{\mathbf{A}}\;\mathit{\mathbf{x}}}_{i\;} 
% ={\mathit{\mathbf{b}}}_{\mathit{\mathbf{i}}}$*,* $i=1\ldotp \ldotp \ldotp s$, 
% where $s<<n$. The right hand side vector changes f for different values of $i$. 
% You need to solve for for only two entries in the unknown vector ${\mathit{\mathbf{x}}}_{\mathit{\mathbf{i}}\;}$. 
% You can assume that you know the indexes of those entries. Provide a pseudo-code 
% of the algorithm such that the CPU cost (the number of floating points operations) 
% is minimum.
% 
% Comment on the cost of your algorithm in terms of the number of Forward/Backward 
% substritutions, and LU decomopositions.
% 
% _*This is assuming that the indices p and q are the two entries we're looking 
% for*_
% 
% _*This pseudocode is inspired by the adjoint method taught in tutorial 4*_
% 
% *Algorithm pseudocode:*
% 
% 
% 
% *function SolveForTwoEntries(A, B, p, q)*
% 
% *//create your selector vectors*
% 
% *d_p = ZeroVector(n)*
% 
% *d_p[p] = 1*
% 
% 
% 
% *d_q = ZeroVector(n)*
% 
% *d_q[q] = 1*
% 
% 
% 
% *//compute transpose and LU decomposition (once)*
% 
% *A_T = transp(A)*
% 
% *[L, U, P] = PLU(A_T)*
% 
% 
% 
% *//solve adjoint systems (once)*
% 
% *//first adjoint: A^T x_a,p = d_p*
% 
% *Pd_p = P * d_p*
% 
% *y_p = FwdSub(L, Pd_p)*
% 
% *x_a_p = BwdSub(U, y_p)*
% 
% 
% 
% *//second adjoint: A^T x_a,q = d_q*
% 
% *Pd_q = P * d_q*
% 
% *y_q = FwdSub(L, Pd_q)*
% 
% *x_a_q = BwdSub(U, y_q)*
% 
% 
% 
% *result_p = EmptyVector(s)*
% 
% *result_q = EmptyVector(s)*
% 
% 
% 
% *//apply adjoint to each right-hand side*
% 
% *for i = 1 to s do*
% 
% *//compute entries via dot products*
% 
% *result_p[i] = DotProduct(x_a_p, B[:,i])*
% 
% *result_q[i] = DotProduct(x_a_q, B[:,i])*
% 
% *end for*
% 
% 
% 
% *return result_p, result_q*
% 
% *end function*
% 
% 
% 
% *To solve for only two specific entries (p and q) in A*x_i = b_i when s<<n, 
% we use the adjoint method which significantly reduces the computational cost 
% per right-hand side.* 
% 
% *Approach:*
%% 
% * *we want to compute adjoint vectors that allow us to find specific entries 
% directly instead of solving the complete system*
% * *transpose A to find its LU decomposition once*
% * *solve two systems to find adjoint vectors corresponding to p and q entries*
% * *for each right hand side, we compute only the needed entries using dot 
% products*
%% 
% *Total cost:*
%% 
% * *LU decomp: O(n^3), done once*
% * *solving for adjoint vectors: 2*O(n^2), done once*
% * *computing dot products: s*O(n) operations, for s right hand sides*
% * *overall complexity: O(n^3+n^2+sn) = O(n^3 + sn) (since the n^2 term gets 
% dominated by the n^3 term in this case for large n)*
%% 
%% Question 5: Cholesky Least Squares Estimation *(10 marks)*
% (a) *Cholesky factorization* is an alternative to LU decompostion which can 
% be applied on symmetric positive definite matrices. A symmetric positive definite 
% matrix$\mathit{\mathbf{M}}$must satisfy following two conditions:
% 
% 1. The matrix $\mathit{\mathbf{M}}$must be symmetric, i.e. $\mathit{\mathbf{M}}={\mathit{\mathbf{M}}}^{\mathit{\mathbf{T}}}$
% 
% 2. The matrix $\mathit{\mathbf{M}}$ must be positive definite,  i.e.  ${\mathit{\mathbf{x}}}^{\mathit{\mathbf{T}}} 
% \mathit{\mathbf{M}}\;\mathit{\mathbf{x}}>0$ for  all  $\mathit{\mathbf{x}}\in 
% \Re^{N\times 1}$, $\left\|\mathit{\mathbf{x}}\right\|\not= 0$.
% 
% If the above two conditons are met then the matrix $\mathit{\mathbf{M}}$ can 
% be factored as 
% 
% $$\mathit{\mathbf{M}}={\mathit{\mathbf{L}}\;\mathit{\mathbf{L}}}^{T\;}$$                 
% 
% where $\mathit{\mathbf{L}}$ is a $N\times N$the lower triangular matrix with 
% real and positive diagonal enteries. The enteries of matrix $\mathit{\mathbf{L}}$ 
% are given by the following expression 
% 
% $$\textbf{L}_{i,j} = \begin{cases}    \sqrt{ \textbf{M}_{j,j} - \sum_{k=1}^{j-1} 
% \textbf{L}_{j,k}^2 }    ,& \text{if  } i =j\\    \frac{1}{\textbf{L}_{j,j}} 
% \left(\textbf{M}_{i,j} - \sum_{k=1}^{j-1} \textbf{L}_{i,k} \textbf{L}_{j,k} 
% \right),              & \text{if  } i>j\\0, & \text{otherwise}\end{cases}$$
% 
% 
% 
% The cost of Cholesky factorization algorithm is rougly half than the LU decompostion. 
% Your task is to implement the Cholesky decomposition alogorithm, the use the 
% outline of the function named _CholeskyDecomposition_ provided in the appendix.
% 
% Use the cell below to test your Cholesky decompostion code by computing the 
% Frobenius norm of  $\left.{\left(\mathit{\mathbf{L}}\right.}^{\;} {\mathit{\mathbf{L}}}^T 
% -M\right)\ldotp$

M= [2 -1 0; -1 2 -1; 0 -1 2];

U = CholeskyDecomposition(M);

Error =  norm(U*U'-M);

%% 
% d) Using Cholesky facotrization scheme $\left({\mathit{\mathbf{A}}}^{\mathit{\mathbf{T}}} 
% \mathit{\mathbf{A}}\right)$ can be factored as ${\mathit{\mathbf{A}}}^{\mathit{\mathbf{T}}} 
% \mathit{\mathbf{A}}\;={\mathit{\mathbf{L}}\;\mathit{\mathbf{L}}}^{T\;}$. This 
% transforms the normal equation into the form,
% 
% $${\mathit{\mathbf{L}}\;\mathit{\mathbf{L}}}^{T\;} \;\mathit{\mathbf{x}}={\mathit{\mathbf{A}}}^{\mathit{\mathbf{T}}\;} 
% \mathit{\mathbf{b}}$$
% 
% The above equation consists of the triangular systems. The solution *x* can 
% be obtained by  first solving $\mathit{\mathbf{L}}\;\mathit{\mathbf{y}}={\mathit{\mathbf{A}}}^{\mathit{\mathbf{T}}\;} 
% \mathit{\mathbf{b}}$ using the forward substitution, then by solving the  ${\mathit{\mathbf{L}}}^{\mathit{\mathbf{T}}} 
% \;\mathit{\mathbf{x}}=\mathit{\mathbf{y}}$ using backward subsitution. Implement 
% the Cholesky solver function named _CholeskySolver_  for least squares in the 
% function in the appendix.
% 
% Test the your Cholesky Least Saquare solver below.

A = rand(30,30); 
b = ones(30,1)

x = CholeskySolver(A,b);

residual = A*x-b;

Norm_residual = norm(residual);

disp(['Question 5d: The norm of the residual after solving using Cholesky based solver is '  num2str(Norm_residual)]);
%% 
% 
%% Appendix
% 
% Question 2 code

function x = forward_sub(L, b)
%pretty much taking slide 24 in lecture 3 and converting it into code

n = length(b);
%solution vector
x = zeros(n,1);

for i = 1:n
    %subtract the known contributions from previous entries:
    %L(i,1:i-1) * x(1:i-1) computes the dot product for row i
    x(i) = (b(i) - L(i,1:i-1)*x(1:i-1)) / L(i,i);
end

end


function x = backward_sub(U, b)
%pretty much taking slide 25 in lecture 3 and converting it into code

n = length(b);
x = zeros(n,1);

%we go from the bottom row (i = n) up to the top row (i = 1)
for i = n:-1:1
    %subtract the contributions from already-solved x(j) with j>i
    % L(i,i+1:n)*x(i+1:n) is the dot product for row i from columns i+1 to n
    x(i) = (b(i) - U(i,i+1:n)*x(i+1:n)) / U(i,i);
end

end

%% 
% 
% Question 3 Code

function [L, U] = myLU(A)



[n, m] = size(A);
%make sure our matrix is square
if n ~= m
    error('Matrix A must be square.');
end

%initialize L as the identity matrix and U as a copy of A
L = eye(n);
U = A;

%perform Gaussian elimination without pivoting
for k = 1:n-1
    if U(k,k) == 0
        error('Zero pivot encountered. Factorization fails without pivoting.');
    end
    for i = k+1:n
        %compute the multiplier for row i
        L(i,k) = U(i,k) / U(k,k);
        %update row i of U
        U(i, k:n) = U(i, k:n) - L(i,k) * U(k, k:n);
    end
end

end


function [L, U, P] = myPLU(A)


[n, m] = size(A);
if n ~= m
    error('Matrix A must be square.');
end

%initialize P as the identity matrix
P = eye(n);
%initialize L as a zero matrix and U as a copy of A
L = zeros(n);
U = A;

%perform Gaussian elimination with partial pivoting
for k = 1:n-1
    %find the pivot row index (largest magnitude in column k, rows k:n)
    [~, pivot_index] = max(abs(U(k:n, k)));
    pivot_index = pivot_index + k - 1;
    
    %if the pivot row is not the current row, swap rows in U, P, and L
    if pivot_index ~= k
        %swap rows k and pivot_index in U
        U([k, pivot_index], :) = U([pivot_index, k], :);
        %swap rows k and pivot_index in P
        P([k, pivot_index], :) = P([pivot_index, k], :);
        %swap the previously computed multipliers in L (columns 1 to k-1)
        if k > 1
            L([k, pivot_index], 1:k-1) = L([pivot_index, k], 1:k-1);
        end
    end
    
    %check for zero pivot (matrix is singular)
    if U(k, k) == 0
        error('Zero pivot encountered. Matrix may be singular.');
    end
    
    %for each row i below row k, compute the multiplier and eliminate
    for i = k+1:n
        L(i, k) = U(i, k) / U(k, k);
        U(i, k:n) = U(i, k:n) - L(i, k) * U(k, k:n);
    end
end

%set the diagonal elements of L to 1
L = L + eye(n);

end
% Question 5 code

function L = CholeskyDecomposition(M)
    
    [n, ~] = size(M);
    L = zeros(n, n);
    
    for j = 1:n
        %compute diagonal element L(j,j)
        sum_val = 0;
        for k = 1:(j-1)
            sum_val = sum_val + L(j,k)^2;
        end
        
        %must be positive definite!!
        if M(j,j) - sum_val <= 0
            error('Matrix is not positive definite');
        end
        %using given equation in question
        L(j,j) = sqrt(M(j,j) - sum_val);
        
        %compute off-diagonal elements L(i,j) where i > j
        for i = (j+1):n
            sum_val = 0;
            for k = 1:(j-1)
                sum_val = sum_val + L(i,k) * L(j,k);
            end
            %using given equation in question
            L(i,j) = (M(i,j) - sum_val) / L(j,j);
        end
    end
end



function x = CholeskySolver(A, b)

    %form the normal equations components
    ATA = A' * A;       
    ATb = A' * b;   
    
    %Cholesky decomp
    L = CholeskyDecomposition(ATA);
    
    %solve L*y = A^T*b using forward substitution
    y = forward_sub(L, ATb);
    
    %solve L^T*x = y using backward substitution
    x = backward_sub(L', y);
end