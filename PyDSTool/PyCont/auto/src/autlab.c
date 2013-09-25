#include <stdio.h>
#include <string.h>

#define DELETED -99
#define INPUT_FORT7  "fort.27"
#define INPUT_FORT8  "fort.28"
#define OUTPUT_FORT7 "fort.37"
#define OUTPUT_FORT8 "fort.38"
#define MAX_BUF 200

struct solution{
  long position;
  int nrowpr;
  int branch;
  int point;
  int type;
  int label;
  int new_label;
  struct solution *next;
};

typedef struct solution *solutionp;

solutionp parse();

void print(solutionp, int);
void type(int,char *);
int command_loop(solutionp);
void delete(solutionp, int);
void relabel(solutionp, int);
void help();
void write8(solutionp);
void write7(solutionp);

main() {
  solutionp root;
  root = parse();

  while(command_loop(root));
}
void write7(solutionp root) {
  FILE *fpin,*fpout;
  char line[MAX_BUF];
  int label,junk,prefix;
  solutionp current = root;

  fpin = fopen(INPUT_FORT7,"r");
  fpout = fopen(OUTPUT_FORT7,"w");
  fgets(line,MAX_BUF,fpin);
  while(feof(fpin)==0) {
    sscanf(line,"%d",&prefix);
    if(prefix != 0) {
      sscanf(line,"%d %d %d %d",&prefix,&junk,&junk,&label);
      if(label != 0) {
	if(label != current->label) {
	  fprintf(stderr,"WARNING: Label mismatch between fort.27 and fort.28\nFiles may be corrupt!"); 
	}
	if(current->new_label == DELETED) {
	  strncpy(&line[10],"   0   0",8);
	} else {
	  char tmp[9];
	  sprintf(tmp,"%4d%4d",current->type,current->new_label);
	  strncpy(&line[10],tmp,8);
	}
	current = current->next;
      }
    }
    fputs(line,fpout);
    fgets(line,MAX_BUF,fpin);
  }
  fclose(fpin);
  fclose(fpout);
}

void write8(solutionp current) {
  int ibr,ntot,itp,lab,nfpr,isw,ntpl,nar,nrowpr,ntst,ncol,npar1;
  int i;
  char line[MAX_BUF];
  FILE *fpin,*fpout;

  fpin = fopen(INPUT_FORT8,"r");
  fpout = fopen(OUTPUT_FORT8,"w");
  while(current != NULL) {
    if(current->new_label != DELETED) {
      rewind(fpin);
      fseek(fpin,current->position,SEEK_SET);
      fscanf(fpin,"%d %d %d %d %d %d %d %d %d %d %d %d",
	     &ibr,&ntot,&itp,&lab,&nfpr,&isw,&ntpl,&nar,
	     &nrowpr,&ntst,&ncol,&npar1);
      fprintf(fpout,"%5d%5d%5d%5d%5d%5d%5d%5d%7d%5d%5d%5d\n",
	      ibr,ntot,itp,current->new_label,nfpr,isw,ntpl,nar,
	      nrowpr,ntst,ncol,npar1);
      /*Go to end of line*/
      while(fgetc(fpin)!='\n');
      for(i=0;i<current->nrowpr;i++) {
	fgets(line,MAX_BUF,fpin);
	fputs(line,fpout);
      }
    }
    current = current->next;
  }
  fclose(fpin);
  fclose(fpout);
}



void delete(solutionp current, int label) {
  while(current != NULL) {
    if((label < 0 || current->label == label) && current->new_label != DELETED) {
      /*Only ask for confirmation if we are deleting everything*/
      if(label < 0) {
	printf("Delete Label %d? (y/n) ",current->label);
	if(fgetc(stdin) == 'y')
	  current->new_label = DELETED;
	/*strip the carriage return */
	fgetc(stdin);
      } else {	
	current->new_label = DELETED;
      }
    }
    current = current->next;
  }
}

void relabel(solutionp current, int label) {
  int index = 1;
  while(current != NULL) {
    if(label < 0 || current->label == label) {
      if(label < 0) {
	if(current->new_label != DELETED) {
	  current->new_label = index;
	  index++;
	}
      } else {
	if(current->new_label != DELETED) {
	  printf("Enter new label for point with label %d: ",current->label);
	  scanf("%d",&(current->new_label));
	  /*strip the carriage return */
	  fgetc(stdin);
	}
      }
    }
    current = current->next;
  }
}

void expandtoken(char *token,int *start, int *end) {
  char *pointer;
  pointer = strstr(token,"-");
  if(pointer == NULL) {
    *start = atoi(token);
    *end   = atoi(token);
  } else {
    *start = atoi(token);
    *end   = atoi(pointer+1);
  }
}

int command_loop(solutionp root) {
  char command[MAX_BUF];
  char *token;
  int label;
  int start,end,i;
  printf("Enter Command : ");
  fgets(command,MAX_BUF,stdin);
  switch(command[0]) {
  case 'l':
    printf("  BR    PT  TY LAB  NEW\n");
    token = strtok(&command[1]," \n");
    if(token != NULL) {
      /*If there are further arguements use them*/
      expandtoken(token,&start,&end);
      for(i=start;i<=end;i++){
	print(root, i);
      }
      token = strtok(NULL," ");
      while(token != NULL) {
	expandtoken(token,&start,&end);
	for(i=start;i<=end;i++){
	  print(root, i);
	}
	token = strtok(NULL," ");
      }
    } else {
      /*Otherwise print them all*/
      print(root,-1);
    }
    break;
  case 'd':
    token = strtok(&command[1]," \n");
    if(token != NULL) {
      /*If there are further arguements use them*/
      expandtoken(token,&start,&end);
      for(i=start;i<=end;i++){
	delete(root, i);
      }
      token = strtok(NULL," ");
      while(token != NULL) {
	expandtoken(token,&start,&end);
	for(i=start;i<=end;i++){
	  delete(root, i);
	}
	token = strtok(NULL," ");
      }
    } else {
      /*Otherwise delete them all*/
      delete(root,-1);
    }
    break;
  case 'r':
    token = strtok(&command[1]," \n");
    if(token != NULL) {
      /*If there are further arguements use them*/
      expandtoken(token,&start,&end);
      for(i=start;i<=end;i++){
	relabel(root, i);
      }
      token = strtok(NULL," ");
      while(token != NULL) {
	expandtoken(token,&start,&end);
	for(i=start;i<=end;i++){
	  relabel(root, i);
	}
	token = strtok(NULL," ");
      }
    } else {
      /*Otherwise relabel them all*/
      relabel(root,-1);
    }
    break;
  case 'w':
    write8(root);
    write7(root);
    printf("Writing files...\n");
    return 0;
    break;
  case 'q':
    return 0;
    break;
  case 'h':
    help();
    break;
  default:
    printf("Invalid Command.  Type 'h' for help.\n");
  }
  return 1;
}

void help() {
  printf(" Available commands :\n");
  printf("   l  :  list labels\n");
  printf("   d  :  delete labels\n");
  printf("   r  :  relabel\n");
  printf("   w  :  rewrite files\n");
  printf("   q  :  quit\n");
  printf("   h  :  help\n");
  printf(" The l, d, and r commands can be followed on the\n");
  printf(" same line by a list of labels, for example,\n");
  printf(" l 13        (list label 13)\n");
  printf(" d 7 13      (delete labels 7 and 13)\n");
  printf(" r 1 12      (relabel 1 and 12)\n");
  printf(" If a list is not specified then the actions are\n");
  printf(" l           (list all labels)\n");
  printf(" d           (delete/confirm all labels)\n");
  printf(" r           (automatic relabeling)\n");
}



void print(solutionp current,int label) {
  while(current!=NULL) {
    char tmp[3];
    type(current->type,tmp);
    if(label < 0 || label == current->label) {
      if(current->new_label != DELETED) {
	printf("%4d%6d  %s%4d%5d\n",
	       current->branch,
	       current->point,
	       tmp,
	       current->label,
	       current->new_label);
      } else {
	printf("%4d%6d  %s%4d  DELETED\n",
	       current->branch,
	       current->point,
	       tmp,
	       current->label);
      }
    }
    current = current->next;
  }
}

void type(int type,char *output) {
  switch(type) {
  case 1:
    strcpy(output,"BP");
    break;
  case 2:
    strcpy(output,"LP");
    break;
  case 3:
    strcpy(output,"HB");
    break;
  case 4:
    strcpy(output,"  ");
    break;
  case -4:
    strcpy(output,"UZ");
    break;
  case 5:
    strcpy(output,"LP");
    break;
  case 6:
    strcpy(output,"BP");
    break;
  case 7:
    strcpy(output,"PD");
    break;
  case 8:
    strcpy(output,"TR");
    break;
  case 9:
    strcpy(output,"EP");
    break;
  case -9:
    strcpy(output,"MX");
    break;
  default:
    strcpy(output,"  ");
  }      
}

solutionp parse() {
  solutionp root = NULL;
  solutionp current = NULL;
  solutionp last = NULL;
  int position,i;
  int ibr,ntot,itp,lab,nfpr,isw,ntpl,nar,nrowpr,ntst,ncol,npar1;
  FILE *fpin;

  fpin = fopen(INPUT_FORT8,"r");
  position = ftell(fpin);
  fscanf(fpin,"%d %d %d %d %d %d %d %d %d %d %d %d",
	 &ibr,&ntot,&itp,&lab,&nfpr,&isw,&ntpl,&nar,
	 &nrowpr,&ntst,&ncol,&npar1);
  while(!feof(fpin)) {
    /* Allocate a new solution*/
    current = (solutionp)malloc(sizeof(struct solution));

    /* If root is NULL that means this is the first node in the list*/
    if(root == NULL)
      root = current;

    /* If last in non-NULL we are not the first one in the list*/
    if(last != NULL)
      last->next = current;

    current->position  = position;
    current->nrowpr    = nrowpr;
    current->branch    = ibr;
    current->point     = ntot;
    current->type      = itp;
    current->label     = lab;
    current->new_label = lab;
    current->next = NULL;

    last = current;

    /*skip the rest of the data by skipping nrowpr + 1 lines (for header)*/
    for(i=0;i<nrowpr+1;i++)
      while(fgetc(fpin)!='\n');

    position = ftell(fpin);
    fscanf(fpin,"%d %d %d %d %d %d %d %d %d %d %d %d",
	   &ibr,&ntot,&itp,&lab,&nfpr,&isw,&ntpl,&nar,
	   &nrowpr,&ntst,&ncol,&npar1);
  }
  fclose(fpin);
  return root;
}



