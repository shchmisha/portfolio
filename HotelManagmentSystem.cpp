#include <iostream>
#include "cstdio"
#include "cstring"
#include "iomanip"

struct student{
    char first_name[50], last_name[50];
    char course[100];
    int section;
};


using namespace std;

int main() {
    FILE *fp, *ft;
    char another, choice;

    struct student e;
    char xfirst_name[50], xlast_name[50];
    long int recsize;

    fp = fopen("users.txt", "rb+");

    if (fp == NULL){
        fp = fopen("users.txt", "wb+");

        if (fp == NULL){
            puts("cannot open file");
            return 0;
        }
    }
    recsize = sizeof(e);
    bool cont = true;
    while(1){
        system("clear");
        cout<<"\t\t====== STUDENT DATABASE MANAGEMENT SYSTEM ======";
        cout<<"\n\n";
        cout<<"\n\n";
        cout<<"\n \t\t\t 1. Add Records";
        cout<<"\n \t\t\t 2. List Records";
        cout<<"\n \t\t\t 3. Modify Records";
        cout<<"\n \t\t\t 4. Delete Records";
        cout<<"\n \t\t\t 5. Exit Program";
        cout<<"\n\n";
        cout<<"\t\t\t Select Your Choice :=> ";
        fflush(stdin);
        cin>>choice;
        switch (choice) {
            case '1':
                fseek(fp, 0, SEEK_END);
                another = 'Y';
                while (another == 'Y' | another == 'y'){
                    system("clear");
                    cout<<"Enter the First Name: ";
                    cin>>e.first_name;
                    cout<<"Enter the Last Name: ";
                    cin>>e.last_name;
                    cout<<"Enter the Course: ";
                    cin>>e.course;
                    cout<<"Enter the Section: ";
                    cin>>e.section;
                    fwrite(&e, recsize, 1, fp);
                    cout<<"\n Add Another Record (Y/N) ";
                    fflush(stdin);
                    another = getchar();
                }
                break;

            case '2':
                system("clear");
                rewind(fp);
                cout << "=== View the Records in the Database ===";
                cout << "\n";
                while (fread(&e, recsize, 1, fp) == 1){
                    cout<<"\n";
                    cout<<"\n"<<e.first_name<<setw(10)<<e.last_name;
                    cout<<"\n";
                    cout<<"\n"<<e.course<<setw(8)<<e.section;
                }
                cout<<"\n\n";
                break;
            case '3':
                system("clear");
                another = 'Y';
                while (another == 'Y' | another == 'y'){
                    cout<<"\n Enter the last Name of Student: ";
                    cin>>xlast_name;
                    rewind(fp);
                    while(fread(&e, recsize, 1, fp)==1){
                        if (strcmp(e.last_name, xlast_name)==0){
                            cout<<"\n Enter new First Name: ";
                            cin>>e.first_name;
                            cout<<"\n Enter new Last Name: ";
                            cin>>e.last_name;
                            cout<<"\n Enter new Course: ";
                            cin>>e.course;
                            cout<<"\n Enter new Section: ";
                            cin>>e.section;
                            fseek(fp, -recsize, SEEK_CUR);
                            fwrite(&e, recsize, 1, fp);
                            break;
                        } else {
                            cout<<"record not found";
                        }
                    }
                    cout<<"\n Modify Another Record (Y/N) ";
                    fflush(stdin);
                    another = getchar();
                }
                break;
            case '4':
                system("clear");
                another = 'Y';
                while(another == 'Y' | another== 'y'){
                    cout<<"\n Enter last name of student to delete: ";
                    cin>>xlast_name;

                    ft = fopen("temp.dat", "wb");
                    rewind(fp);

                    while (fread(&e, recsize, 1, fp)==1) {
                        if (strcmp(e.last_name, xlast_name) != 0) {
                            fwrite(&e, recsize, 1, ft);
                        }
                    }
                    fclose(fp);
                    fclose(ft);
                    remove("users.txt");
                    rename("temp.dat", "users.txt");

                    fp = fopen("users.txt", "rb+");
                    cout<<"\n Delete Another Record (Y/N) ";
                    fflush(stdin);
                    cin>>another;
                }
                break;
            case '5':
                fclose(fp);
                exit(0);
            default:
                break;
        }
    }
    return 0;
}
