# This function assumes 1 initial infective and N-1 initially susceptibles
# Per-person infection rate is beta/N

simSIR.Markov <- function(N, beta, gamma) {
  
  # initial number of infectives and susceptibles;
  I <- 1
  S <- N-1;
  
  # recording time;
  t <- 0;
  times <- c(t);
  
  # a vector which records the type of event (1=infection, 2=removal)
  type <- c(1);
  
  while (I > 0) {
    
    # time to next event;
    t <- t + rexp(1, (beta/N)*I*S + gamma*I);
    times <- append(times, t);
    
    if (runif(1) < beta*S/(beta*S + N*gamma)) {
      # infection
      I <- I+1;
      S <- S-1;
      type <- append(type, 1);
    }
    else {
      #removal
      I <- I-1
      type <- append(type, 2);
    }
  }
  
  # add the following lines in each of the simulation functions
  final.size <- sum(type==1) - 1
  duration <- max(times)
  
  # modify the existing `res` objetct to store the final size and the duration
  
  res <- list("t"=times, "type"=type, "final.size"=final.size, "duration" = duration);
  res
}